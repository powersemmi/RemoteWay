//! `RemoteWay` server binary: screen capture, compression, and transport over SSH.
//!
//! Captures the Wayland screen via wlr-screencopy/ext-image-capture/portal,
//! compresses frames with delta+LZ4/Zstd, and streams them to the client
//! over stdin/stdout (launched by the client via SSH). Also receives and
//! injects input events from the client.

use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use anyhow::{Context, Result};
use clap::Parser;
use mimalloc::MiMalloc;
use remoteway_capture::error::CaptureError;
use remoteway_capture::thread::CaptureThreadConfig;
use remoteway_input::inject_thread::InputInjectConfig;
use remoteway_transport::ssh_transport::SshTransport;
use tracing::info;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

mod cli;
mod pipeline;

#[cfg(feature = "tracy")]
use tracy_client::Client as TracyClient;

fn main() -> Result<()> {
    // SAFETY: mlockall with MCL_CURRENT|MCL_FUTURE prevents page faults on hot path.
    // Must be called before any pipeline threads are spawned.
    let ret = unsafe { libc::mlockall(libc::MCL_CURRENT | libc::MCL_FUTURE) };
    if ret != 0 {
        eprintln!(
            "mlockall failed (errno {}): run as root or set RLIMIT_MEMLOCK",
            // SAFETY: reading errno immediately after failed syscall is safe.
            unsafe { *libc::__errno_location() }
        );
    }

    #[cfg(feature = "tracy")]
    let _tracy = TracyClient::start();

    let cli = cli::Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env().add_directive(
                "remoteway=info"
                    .parse()
                    .context("BUG: hardcoded directive must parse")?,
            ),
        )
        .with_writer(std::io::stderr)
        .init();

    info!("remoteway-server starting");

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .context("failed to build tokio runtime")?;

    rt.block_on(run(cli))
}

/// Launch the child process with `WAYLAND_DISPLAY` propagation.
///
/// stdin/stdout are redirected to null so the child does not inherit the
/// SSH transport pipes (prevents protocol corruption and ensures SSH
/// closes promptly when the server exits).
fn launch_child(command: &[String]) -> Result<tokio::process::Child> {
    let mut cmd = tokio::process::Command::new(&command[0]);
    let _ = cmd.args(&command[1..]);
    let _ = cmd.stdin(Stdio::null());
    let _ = cmd.stdout(Stdio::null());
    let _ = cmd.stderr(Stdio::inherit());
    match std::env::var("WAYLAND_DISPLAY") {
        Ok(wl_display) => {
            let _ = cmd.env("WAYLAND_DISPLAY", &wl_display);
            info!(wayland_display = %wl_display, "propagating WAYLAND_DISPLAY to child");
        }
        Err(_) => {
            // WAYLAND_DISPLAY not set; child will inherit whatever the
            // compositor provides or fall back to its own discovery.
        }
    }
    let child = cmd
        .spawn()
        .with_context(|| format!("failed to launch: {}", command[0]))?;
    info!(pid = ?child.id(), cmd = ?command, "child process launched");
    Ok(child)
}

/// Retry loop: waits up to 5 s for a capture backend to succeed.
///
/// Bails immediately on permanent errors (missing protocol).
fn retry_capture_backend(
    try_fn: impl FnMut() -> Result<Box<dyn remoteway_capture::backend::CaptureBackend>>,
    what: &str,
) -> Result<Box<dyn remoteway_capture::backend::CaptureBackend>> {
    retry_capture_backend_with_params(try_fn, what, 50, std::time::Duration::from_millis(100))
}

/// Core retry logic parametrised for testability.
fn retry_capture_backend_with_params(
    mut try_fn: impl FnMut() -> Result<Box<dyn remoteway_capture::backend::CaptureBackend>>,
    what: &str,
    max_retries: u32,
    retry_interval: std::time::Duration,
) -> Result<Box<dyn remoteway_capture::backend::CaptureBackend>> {
    for attempt in 1..=max_retries {
        match try_fn() {
            Ok(backend) => return Ok(backend),
            Err(e) => {
                // Permanent failures (missing protocol) — bail immediately.
                if matches!(
                    e.root_cause().downcast_ref::<CaptureError>(),
                    Some(ce) if !ce.is_retriable()
                ) {
                    return Err(e);
                }
                if attempt == max_retries {
                    return Err(e).context(format!("{what} not found after 5 s"));
                }
                if attempt % 10 == 0 {
                    info!(attempt, "waiting for {what}...");
                }
                std::thread::sleep(retry_interval);
            }
        }
    }
    unreachable!()
}

/// Determine capture mode from CLI arguments.
///
/// Returns `(auto_detect_child, explicit_app_id_launch)`:
/// - `auto_detect_child`: command present, no `--app-id` -> auto-detect child window via toplevel diff.
/// - `explicit_app_id_launch`: command present + `--app-id` -> launch child, find by `app_id`.
/// - Both `false`: no command -> capture by `--app-id`, `--output`, or default output.
fn determine_capture_mode(app_id: Option<&str>, command: &[String]) -> (bool, bool) {
    let auto_detect_child = app_id.is_none() && !command.is_empty();
    let explicit_app_id_launch = app_id.is_some() && !command.is_empty();
    (auto_detect_child, explicit_app_id_launch)
}

async fn run(cli: cli::Cli) -> Result<()> {
    let shutdown = Arc::new(AtomicBool::new(false));

    #[cfg(feature = "portal")]
    if cli.select_source {
        info!("opening portal source-selection dialog…");
        remoteway_capture::desktop_detect::ensure_wayland_env();
        remoteway_capture::portal::PortalBackend::setup_restore_token()
            .context("portal source selection failed")?;
        info!("restore token saved; SSH sessions can now skip the dialog");
        return Ok(());
    }

    // Transport over stdin/stdout (server is launched by client via SSH).
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();
    let (mut transport, _io_handle) = SshTransport::new(stdin, stdout);
    let sender = transport.sender();

    // Send server handshake.
    let handshake_data = pipeline::build_handshake(&cli.capture, &cli.compress);
    // INTENTIONAL: send_anchor always succeeds on a freshly-created transport.
    let _ = sender.send_anchor(handshake_data);
    info!("handshake sent, waiting for client...");

    // Wait for client handshake.
    if let Some(msg) = transport.recv().await {
        if matches!(
            msg.header.msg_type(),
            Ok(remoteway_proto::header::MsgType::Handshake)
        ) {
            info!("client handshake received");
        } else {
            anyhow::bail!("expected handshake, got msg_type={}", {
                msg.header.msg_type
            });
        }
    } else {
        anyhow::bail!("transport closed before handshake");
    }

    // Ensure WAYLAND_DISPLAY is available (may need discovery when launched via SSH).
    if !remoteway_capture::desktop_detect::ensure_wayland_env() {
        anyhow::bail!("WAYLAND_DISPLAY not found: no running Wayland compositor detected");
    }

    // Determine capture mode and launch strategy.
    let (auto_detect_child, explicit_app_id_launch) =
        determine_capture_mode(cli.app_id.as_deref(), &cli.command);

    // Snapshot existing toplevels before spawning (for auto-detect mode).
    let known_toplevels = if auto_detect_child {
        match remoteway_capture::detect::enumerate_toplevels() {
            Ok(list) => {
                info!(
                    count = list.len(),
                    "snapshot of existing toplevels taken for child detection"
                );
                Some(list)
            }
            Err(_) => {
                info!(
                    "toplevel enumeration not available, \
                     will capture full output instead of child window"
                );
                None
            }
        }
    } else {
        None
    };

    // Launch child early if we need its window to exist before capture init.
    let mut child: Option<tokio::process::Child> = if auto_detect_child || explicit_app_id_launch {
        Some(launch_child(&cli.command)?)
    } else {
        None
    };

    // Create capture backend.
    let capture_arg = cli.capture.clone();
    let backend = if auto_detect_child {
        if let Some(ref known) = known_toplevels {
            let ids: Vec<String> = known.iter().map(|t| t.identifier.clone()).collect();
            match retry_capture_backend(
                || pipeline::create_capture_backend_for_child(&capture_arg, &ids),
                "child window",
            ) {
                Ok(backend) => backend,
                Err(e) => {
                    tracing::warn!(
                        "per-window capture via ext-image-capture unavailable ({e:#}), falling back to full-screen"
                    );
                    pipeline::create_capture_backend(&cli.capture, cli.output.as_deref(), None)?
                }
            }
        } else {
            // Fallback: no toplevel protocol → capture full output.
            pipeline::create_capture_backend(&cli.capture, cli.output.as_deref(), None)?
        }
    } else if explicit_app_id_launch {
        let app_id = cli.app_id.as_deref();
        retry_capture_backend(
            || pipeline::create_capture_backend(&capture_arg, None, app_id),
            "toplevel window",
        )?
    } else {
        pipeline::create_capture_backend(
            &cli.capture,
            cli.output.as_deref(),
            cli.app_id.as_deref(),
        )?
    };
    info!(backend = backend.name(), "capture backend ready");

    // Spawn capture thread (Core 1, SCHED_FIFO 90).
    let capture_config = CaptureThreadConfig::default();
    let capture = remoteway_capture::thread::CaptureThread::spawn(backend, capture_config)
        .context("failed to spawn capture thread")?;
    info!("capture thread started");

    // Spawn input inject thread (Core 0, SCHED_FIFO 99).
    let inject_config = InputInjectConfig::default();
    let input_inject = remoteway_input::inject_thread::InputInjectThread::spawn(inject_config)
        .context("failed to spawn input inject thread")?;
    info!("input inject thread started");

    // Shared target resolution (packed: width << 32 | height). 0 = native.
    let target_resolution = Arc::new(AtomicU64::new(0));

    // Spawn compress+send thread (Core 2).
    let compress_handle = pipeline::spawn_compress_thread(
        capture,
        sender,
        cli.compress.clone(),
        shutdown.clone(),
        target_resolution.clone(),
    )?;
    info!("compress-send thread started");

    // Launch target application if not already launched above.
    if !auto_detect_child && !explicit_app_id_launch && !cli.command.is_empty() {
        child = Some(launch_child(&cli.command)?);
    }

    // Receive loop: dispatch incoming messages (input events from client).
    // Also handles shutdown on transport disconnect.
    let recv_shutdown = shutdown.clone();
    let recv_target_res = target_resolution.clone();
    let recv_task = tokio::spawn(async move {
        pipeline::recv_dispatch_loop(&mut transport, input_inject, recv_shutdown, recv_target_res)
            .await;
    });

    // Wait for shutdown signal or child process exit.
    let signal_shutdown = shutdown.clone();
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {
            info!("SIGINT received, shutting down");
            signal_shutdown.store(true, Ordering::Release);
        }
        _ = recv_task => {
            info!("transport closed, shutting down");
            shutdown.store(true, Ordering::Release);
        }
        _ = async {
            if let Some(ref mut c) = child {
                if let Err(e) = c.wait().await {
                    tracing::warn!(error = %e, "error waiting for child process");
                }
            } else {
                std::future::pending::<()>().await;
            }
        } => {
            info!("child process exited, shutting down");
            shutdown.store(true, Ordering::Release);
        }
    }

    // Wait for compress thread to finish.
    shutdown.store(true, Ordering::Release);
    if let Err(e) = compress_handle.join() {
        tracing::warn!(
            error = ?e,
            "compress thread panicked"
        );
    }

    // Kill child if still running.
    if let Some(ref mut c) = child {
        if let Err(e) = c.start_kill() {
            tracing::warn!(error = %e, "failed to send SIGKILL to child");
        }
        if let Err(e) = c.wait().await {
            tracing::warn!(error = %e, "error waiting for child after kill");
        }
    }

    info!("remoteway-server stopped");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use remoteway_capture::backend::CaptureBackend;
    use remoteway_capture::backend::CapturedFrame;
    use remoteway_capture::error::CaptureError;
    use std::cell::Cell;
    use std::time::Duration;

    /// Minimal mock backend returned by successful `retry_capture_backend` calls.
    struct MockBackend;

    impl CaptureBackend for MockBackend {
        fn next_frame(&mut self) -> Result<CapturedFrame, CaptureError> {
            Err(CaptureError::SessionEnded)
        }

        fn name(&self) -> &'static str {
            "mock"
        }

        fn stop(&mut self) {}
    }

    // ── retry_capture_backend tests ──────────────────────────────────────

    #[test]
    fn retry_succeeds_immediately() {
        let result = retry_capture_backend_with_params(
            || Ok(Box::new(MockBackend) as Box<dyn CaptureBackend>),
            "test",
            5,
            Duration::ZERO,
        );
        assert!(result.is_ok());
        assert_eq!(result.unwrap().name(), "mock");
    }

    #[test]
    fn retry_fails_then_succeeds() {
        let attempt = Cell::new(0u32);
        let result = retry_capture_backend_with_params(
            || {
                let n = attempt.get();
                attempt.set(n + 1);
                if n < 3 {
                    // Retriable error (CaptureFailed).
                    Err(CaptureError::CaptureFailed("not ready yet".into()).into())
                } else {
                    Ok(Box::new(MockBackend) as Box<dyn CaptureBackend>)
                }
            },
            "test",
            10,
            Duration::ZERO,
        );
        assert!(result.is_ok());
        assert_eq!(attempt.get(), 4); // 3 failures + 1 success
    }

    #[test]
    fn retry_bails_on_non_retriable_error() {
        let attempt = Cell::new(0u32);
        let result = retry_capture_backend_with_params(
            || {
                attempt.set(attempt.get() + 1);
                // NoBackend is a permanent (non-retriable) error.
                Err(CaptureError::NoBackend.into())
            },
            "test",
            50,
            Duration::ZERO,
        );
        assert!(result.is_err());
        // Must bail on the very first attempt — no retries.
        assert_eq!(attempt.get(), 1);
        let msg = format!("{:#}", result.err().unwrap());
        assert!(
            msg.contains("no suitable capture protocol"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn retry_exhausts_all_attempts() {
        let attempt = Cell::new(0u32);
        let result = retry_capture_backend_with_params(
            || {
                attempt.set(attempt.get() + 1);
                Err(CaptureError::CaptureFailed("transient".into()).into())
            },
            "child window",
            5,
            Duration::ZERO,
        );
        assert!(result.is_err());
        assert_eq!(attempt.get(), 5);
        let msg = format!("{:#}", result.err().unwrap());
        assert!(
            msg.contains("not found after 5 s"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn retry_bails_on_session_ended() {
        let attempt = Cell::new(0u32);
        let result = retry_capture_backend_with_params(
            || {
                attempt.set(attempt.get() + 1);
                Err(CaptureError::SessionEnded.into())
            },
            "test",
            10,
            Duration::ZERO,
        );
        assert!(result.is_err());
        assert_eq!(attempt.get(), 1);
    }

    #[test]
    fn retry_bails_on_output_not_found() {
        let attempt = Cell::new(0u32);
        let result = retry_capture_backend_with_params(
            || {
                attempt.set(attempt.get() + 1);
                Err(CaptureError::OutputNotFound("DP-1".into()).into())
            },
            "test",
            10,
            Duration::ZERO,
        );
        assert!(result.is_err());
        assert_eq!(attempt.get(), 1);
    }

    #[test]
    fn retry_non_capture_error_is_retried() {
        // An anyhow error that is NOT a CaptureError should be retried
        // (the downcast to CaptureError will fail, so it is treated as retriable).
        let attempt = Cell::new(0u32);
        let result = retry_capture_backend_with_params(
            || {
                let n = attempt.get();
                attempt.set(n + 1);
                if n < 2 {
                    Err(anyhow::anyhow!("transient I/O error"))
                } else {
                    Ok(Box::new(MockBackend) as Box<dyn CaptureBackend>)
                }
            },
            "test",
            10,
            Duration::ZERO,
        );
        assert!(result.is_ok());
        assert_eq!(attempt.get(), 3);
    }

    // ── determine_capture_mode tests ─────────────────────────────────────

    #[test]
    fn mode_auto_detect_child() {
        // command present, no app_id => auto-detect child
        let (auto, explicit) = determine_capture_mode(None, &["firefox".to_string()]);
        assert!(auto);
        assert!(!explicit);
    }

    #[test]
    fn mode_explicit_app_id_launch() {
        // command present, app_id present => explicit app_id launch
        let (auto, explicit) =
            determine_capture_mode(Some("org.mozilla.firefox"), &["firefox".to_string()]);
        assert!(!auto);
        assert!(explicit);
    }

    #[test]
    fn mode_no_command_with_app_id() {
        // no command, app_id present => capture by app_id
        let (auto, explicit) = determine_capture_mode(Some("org.mozilla.firefox"), &[]);
        assert!(!auto);
        assert!(!explicit);
    }

    #[test]
    fn mode_no_command_no_app_id() {
        // no command, no app_id => capture default output
        let (auto, explicit) = determine_capture_mode(None, &[]);
        assert!(!auto);
        assert!(!explicit);
    }

    #[test]
    fn mode_multi_arg_command() {
        let (auto, explicit) =
            determine_capture_mode(None, &["firefox".to_string(), "--headless".to_string()]);
        assert!(auto);
        assert!(!explicit);
    }

    // ── launch_child tests ───────────────────────────────────────────────

    #[tokio::test]
    async fn launch_child_true_succeeds() {
        let child = launch_child(&["true".to_string()]);
        assert!(child.is_ok());
        let mut child = child.unwrap();
        let status = child.wait().await.unwrap();
        assert!(status.success());
    }

    #[tokio::test]
    async fn launch_child_echo_succeeds() {
        let child = launch_child(&["echo".to_string(), "hello".to_string()]);
        assert!(child.is_ok());
        let mut child = child.unwrap();
        let status = child.wait().await.unwrap();
        assert!(status.success());
    }

    #[tokio::test]
    async fn launch_child_false_returns_nonzero() {
        let child = launch_child(&["false".to_string()]);
        assert!(child.is_ok());
        let mut child = child.unwrap();
        let status = child.wait().await.unwrap();
        assert!(!status.success());
    }

    #[test]
    fn launch_child_nonexistent_binary_fails() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let result =
            rt.block_on(async { launch_child(&["__nonexistent_binary_12345__".to_string()]) });
        assert!(result.is_err());
    }
}

//! `RemoteWay` client binary: SSH transport, decompression, display, and input capture.
//!
//! Launches remoteway-server on the remote host via SSH, receives compressed
//! frames over stdin/stdout, decompresses and displays them using Wayland SHM
//! buffers, and captures local input events to forward to the server.

use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};
use clap::Parser;
use mimalloc::MiMalloc;
use remoteway_display::DisplayThreadConfig;
use remoteway_input::capture_thread::InputCaptureConfig;
use remoteway_interpolate::{BackendDetector, InterpolationManager};
use remoteway_transport::ssh_transport::SshTransport;
use tracing::{debug, info};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

mod cli;
mod pipeline;

#[cfg(feature = "tracy")]
use tracy_client::Client as TracyClient;

#[allow(clippy::expect_used)]
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

    info!("remoteway-client starting");

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .context("failed to build tokio runtime")?;

    rt.block_on(run(cli))
}

async fn run(cli: cli::Cli) -> Result<()> {
    let shutdown = Arc::new(AtomicBool::new(false));

    // Launch SSH subprocess: ssh [opts] host remoteway-server [-- app]
    let ssh_args = cli.ssh_command();
    info!(cmd = ?ssh_args, "launching SSH");

    let mut child = tokio::process::Command::new(&ssh_args[0])
        .args(&ssh_args[1..])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .with_context(|| format!("failed to launch: {}", ssh_args.join(" ")))?;

    let child_stdin = child.stdin.take().context("failed to capture SSH stdin")?;
    let child_stdout = child
        .stdout
        .take()
        .context("failed to capture SSH stdout")?;

    // Create transport over SSH stdin/stdout.
    let (mut transport, _io_handle) = SshTransport::new(child_stdout, child_stdin);
    let sender = transport.sender();

    // Send client handshake.
    let handshake_data = pipeline::build_handshake();
    if !sender.send_anchor(handshake_data) {
        tracing::warn!("failed to send anchor frame (transport closed)");
    }
    info!("handshake sent, waiting for server...");

    // Wait for server handshake.
    if let Some(msg) = transport.recv().await {
        if matches!(
            msg.header.msg_type(),
            Ok(remoteway_proto::header::MsgType::Handshake)
        ) {
            info!("server handshake received");
        } else {
            anyhow::bail!("expected handshake, got msg_type={}", {
                msg.header.msg_type
            });
        }
    } else {
        anyhow::bail!("transport closed before handshake");
    }

    // Send target resolution if requested.
    if let Some(res) = &cli.resolution {
        let msg = pipeline::build_target_resolution(res.width, res.height);
        if !sender.send_anchor(msg) {
            tracing::warn!("failed to send target resolution frame (transport closed)");
        }
        info!(
            width = res.width,
            height = res.height,
            "target resolution sent"
        );
    }

    // Detect interpolation backend.
    let interpolation = if cli.no_interpolate {
        info!("interpolation disabled");
        None
    } else if let Some(kind) = cli.interpolation_backend {
        info!(
            backend = kind.name(),
            "using user-selected interpolation backend"
        );
        match BackendDetector::create_backend(kind) {
            Ok(backend) => Some(InterpolationManager::new(backend)),
            Err(e) => {
                info!(
                    "failed to create interpolation backend '{}': {e}",
                    kind.name()
                );
                None
            }
        }
    } else {
        let available = BackendDetector::detect_available();
        info!(backends = ?available, "available interpolation backends");
        match BackendDetector::select_best() {
            Ok(backend) => {
                info!(name = backend.name(), "interpolation backend selected");
                Some(InterpolationManager::new(backend))
            }
            Err(e) => {
                info!("no interpolation backend available: {e}");
                None
            }
        }
    };

    // Spawn display thread (Core 3).
    let display_config = DisplayThreadConfig::default();
    let display = remoteway_display::DisplayThread::spawn(display_config)
        .context("failed to spawn display thread")?;
    info!("display thread started");

    // Spawn input capture thread (Core 0, SCHED_FIFO 99).
    let input_sender = pipeline::make_input_sender(sender.clone());
    let input_config = InputCaptureConfig::default();
    let _input_capture =
        remoteway_input::capture_thread::InputCaptureThread::spawn(input_config, input_sender)
            .context("failed to spawn input capture thread")?;
    info!("input capture thread started");

    // Receive + decompress + display loop (runs in tokio task).
    let recv_shutdown = shutdown.clone();
    let recv_task = tokio::spawn(async move {
        pipeline::recv_decompress_loop(&mut transport, display, interpolation, recv_shutdown).await;
    });

    // Wait for shutdown.
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
        status = child.wait() => {
            match status {
                Ok(s) => info!(status = %s, "SSH process exited"),
                Err(e) => info!(error = %e, "SSH process error"),
            }
            shutdown.store(true, Ordering::Release);
        }
    }

    // Kill SSH child if still running.
    if let Err(e) = child.kill().await {
        // Child likely already exited — non-critical during shutdown
        debug!("failed to kill SSH child (already exited?): {e}");
    }

    info!("remoteway-client stopped");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use remoteway_interpolate::backend::BackendKind;

    /// Determines the interpolation strategy based on CLI flags.
    ///
    /// Returns:
    /// - `InterpolationMode::Disabled` if `--no-interpolate` is set
    /// - `InterpolationMode::Explicit(kind)` if `--interpolation-backend` is set
    /// - `InterpolationMode::AutoDetect` otherwise
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum InterpolationMode {
        /// User explicitly disabled interpolation with `--no-interpolate`.
        Disabled,
        /// User selected a specific backend with `--interpolation-backend`.
        Explicit(BackendKind),
        /// Auto-detect the best available backend.
        AutoDetect,
    }

    fn determine_interpolation_mode(
        no_interpolate: bool,
        interpolation_backend: Option<BackendKind>,
    ) -> InterpolationMode {
        if no_interpolate {
            InterpolationMode::Disabled
        } else if let Some(kind) = interpolation_backend {
            InterpolationMode::Explicit(kind)
        } else {
            InterpolationMode::AutoDetect
        }
    }

    #[test]
    fn determine_mode_disabled_takes_priority() {
        let mode = determine_interpolation_mode(true, None);
        assert_eq!(mode, InterpolationMode::Disabled);
    }

    #[test]
    fn determine_mode_disabled_overrides_explicit() {
        let mode = determine_interpolation_mode(true, Some(BackendKind::LinearBlend));
        assert_eq!(mode, InterpolationMode::Disabled);
    }

    #[test]
    fn determine_mode_explicit_when_set() {
        let mode = determine_interpolation_mode(false, Some(BackendKind::LinearBlend));
        assert_eq!(mode, InterpolationMode::Explicit(BackendKind::LinearBlend));
    }

    #[test]
    fn determine_mode_auto_when_unset() {
        let mode = determine_interpolation_mode(false, None);
        assert_eq!(mode, InterpolationMode::AutoDetect);
    }

    #[test]
    fn build_ssh_args_includes_host() {
        let cli = cli::Cli {
            host: "test.example.com".to_string(),
            ssh_opt: Vec::new(),
            command: Vec::new(),
            no_interpolate: false,
            server_bin: "remoteway-server".to_string(),
            compress: cli::CompressArg::Lz4,
            capture: cli::CaptureBackendArg::Auto,
            app_id: None,
            resolution: None,
            interpolation_backend: None,
        };
        let args = cli.ssh_command();
        assert!(args.iter().any(|a| a.as_str().contains("test.example.com")));
    }

    #[test]
    fn interpolation_mode_clone_copy_eq() {
        let m1 = InterpolationMode::Disabled;
        let m2 = m1;
        assert_eq!(m1, m2);
        let m3 = InterpolationMode::AutoDetect;
        assert_ne!(m1, m3);
    }

    #[test]
    fn interpolation_mode_explicit_variants() {
        let m1 = InterpolationMode::Explicit(BackendKind::LinearBlend);
        let m2 = InterpolationMode::Explicit(BackendKind::LinearBlend);
        assert_eq!(m1, m2);
    }

    #[test]
    fn interpolation_mode_debug() {
        let m = InterpolationMode::Disabled;
        let dbg = format!("{:?}", m);
        assert!(dbg.contains("Disabled"));
    }
}

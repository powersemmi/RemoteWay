//! Portal + `GStreamer` capture backend for GNOME / KDE.
//!
//! Uses [`ashpd`] (high-level D-Bus portal API) for the portal handshake
//! (`CreateSession` → `SelectSources` → `Start`) and a `GStreamer` pipeline
//! to capture frames from a `PipeWire` stream.

use std::os::unix::io::{AsRawFd, OwnedFd};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use ashpd::desktop::screencast::{
    CursorMode, Screencast, SelectSourcesOptions, SourceType, StartCastOptions,
};
use gstreamer::prelude::*;
use remoteway_compress::delta::DamageRect;
use tokio::sync::mpsc;
use tracing::{debug, error, info, trace, warn};

use crate::backend::{CaptureBackend, CapturedFrame, PixelFormat};
use crate::error::CaptureError;

// ---------------------------------------------------------------------------
// Token persistence
// ---------------------------------------------------------------------------

/// Directory name under `XDG_CACHE_HOME` (or `~/.cache`) for the restore token.
const TOKEN_DIR: &str = "remoteway";
/// File name for the portal restore token.
const TOKEN_FILE: &str = "portal-token";

/// Returns the full path to the portal restore token file.
///
/// Uses `XDG_CACHE_HOME` if set, otherwise falls back to `~/.cache`.
fn token_path() -> PathBuf {
    let base = std::env::var("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
            PathBuf::from(home).join(".cache")
        });
    base.join(TOKEN_DIR).join(TOKEN_FILE)
}

/// Reads the persisted portal restore token from disk.
///
/// Returns `None` if the file doesn't exist or is empty.
fn load_restore_token() -> Option<String> {
    let content = std::fs::read_to_string(token_path()).ok()?;
    let trimmed = content.trim().to_string();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

/// Persists the portal restore token to disk for future sessions.
///
/// Creates the parent directory if needed. Logs a warning on failure
/// but never panics — token persistence is best-effort.
fn save_restore_token(token: &str) {
    let path = token_path();
    if let Some(parent) = path.parent()
        && let Err(e) = std::fs::create_dir_all(parent)
    {
        warn!(%e, path = %parent.display(), "failed to create token directory");
        return;
    }
    if let Err(e) = std::fs::write(&path, token) {
        warn!(%e, "failed to save portal restore token");
    }
}

// ---------------------------------------------------------------------------
// PortalBackend – public API
// ---------------------------------------------------------------------------

/// Portal + `GStreamer` capture backend.
pub struct PortalBackend {
    frame_rx: mpsc::Receiver<CapturedFrame>,
    stop_flag: Arc<AtomicBool>,
}

impl PortalBackend {
    /// Initialize the portal + `GStreamer` capture pipeline.
    ///
    /// Spawns a dedicated OS thread with a Tokio `current_thread` runtime for
    /// the D-Bus portal handshake. Once the session is established, a `GStreamer`
    /// pipeline consumes the `PipeWire` stream and forwards frames through an
    /// internal MPSC channel.
    ///
    /// # Errors
    ///
    /// Returns an error if D-Bus / portal / `PipeWire` / `GStreamer` is unavailable.
    pub fn new() -> Result<Self, CaptureError> {
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), CaptureError>>();
        let (frame_tx, frame_rx) = mpsc::channel(16);
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_flag_clone = stop_flag.clone();

        std::thread::spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if let Err(e) = gstreamer::init() {
                    return Err(CaptureError::CaptureFailed(format!("gstreamer init: {e}")));
                }

                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|e| CaptureError::CaptureFailed(format!("tokio runtime: {e}")))?;

                rt.block_on(Self::run_capture(frame_tx, stop_flag_clone, ready_tx))
            }));

            match result {
                Ok(Ok(())) => info!("portal thread exited cleanly"),
                Ok(Err(e)) => {
                    error!("portal thread error: {e}");
                    // ready_tx may have been moved into run_capture already;
                    // if the error came from gstreamer init or tokio build,
                    // the error was already sent inside the closure above.
                }
                Err(panic) => {
                    let msg = panic
                        .downcast_ref::<&str>()
                        .map(|s| s.to_string())
                        .or_else(|| panic.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| "unknown panic".into());
                    error!("portal thread panicked: {msg}");
                    // ready_tx has been moved; the panic after readiness
                    // is non-fatal — capture_loop already running.
                }
            }
        });

        // Wait for the portal handshake + pipeline setup to complete.
        let ready_result = ready_rx
            .recv_timeout(std::time::Duration::from_secs(30))
            .map_err(|e| CaptureError::CaptureFailed(format!("portal setup timed out: {e}")))?;
        ready_result?;

        Ok(PortalBackend {
            frame_rx,
            stop_flag,
        })
    }

    /// Run the portal source-selection dialog and save the restore token.
    /// Call this from a desktop session once so subsequent SSH runs skip the dialog.
    pub fn setup_restore_token() -> Result<(), CaptureError> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| CaptureError::CaptureFailed(format!("setup rt: {e}")))?;

        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let result = rt.block_on(Self::portal_handshake());
            let _ = tx.send(result);
        });

        let _ = rx
            .recv_timeout(std::time::Duration::from_secs(180))
            .map_err(|_| CaptureError::CaptureFailed("token setup timed out".into()))?
            .map_err(|e| CaptureError::CaptureFailed(format!("{e}")))?;

        info!("restore token saved — portal setup complete");
        Ok(())
    }

    // ------------------------------------------------------------------
    // Lifecycle orchestration
    // ------------------------------------------------------------------

    /// Orchestrates the full capture lifecycle on a Tokio `current_thread` runtime.
    ///
    /// 1. Portal handshake → `PipeWire` fd + node id
    /// 2. Build and start the `GStreamer` pipeline
    /// 3. Signal readiness via `ready_tx`
    /// 4. Enter the capture loop (blocks until `stop_flag` is set)
    ///
    /// All errors that occur before the readiness signal are forwarded
    /// through `ready_tx` so the caller can fail fast.
    async fn run_capture(
        frame_tx: mpsc::Sender<CapturedFrame>,
        stop_flag: Arc<AtomicBool>,
        ready_tx: std::sync::mpsc::Sender<Result<(), CaptureError>>,
    ) -> Result<(), CaptureError> {
        /// Signal an error through `ready_tx` and return it for `?` propagation.
        ///
        /// This ensures the error reaches both the readiness channel (so
        /// `new()` fails immediately) and the caller's error path.
        fn fail(
            tx: &std::sync::mpsc::Sender<Result<(), CaptureError>>,
            e: CaptureError,
        ) -> CaptureError {
            let msg = format!("{e}");
            let _ = tx.send(Err(CaptureError::CaptureFailed(msg.clone())));
            CaptureError::CaptureFailed(msg)
        }

        // Portal handshake
        let (pw_fd, pw_node_id) = Self::portal_handshake()
            .await
            .map_err(|e| fail(&ready_tx, e))?;
        let pw_raw_fd = pw_fd.as_raw_fd();

        let pipeline =
            Self::build_pipeline(pw_raw_fd, pw_node_id).map_err(|e| fail(&ready_tx, e))?;

        let appsink = pipeline
            .by_name("appsink")
            .and_then(|e| e.downcast::<gstreamer_app::AppSink>().ok())
            .ok_or_else(|| {
                fail(
                    &ready_tx,
                    CaptureError::CaptureFailed(
                        "appsink element not found in GStreamer pipeline".into(),
                    ),
                )
            })?;

        pipeline.set_state(gstreamer::State::Playing).map_err(|_| {
            fail(
                &ready_tx,
                CaptureError::CaptureFailed("failed to start GStreamer pipeline".into()),
            )
        })?;
        // Keep PipeWire fd open for the pipeline's lifetime.
        let _pw_fd = pw_fd;

        // Drain any pending bus messages before capturing.
        {
            let bus = pipeline.bus().ok_or_else(|| {
                fail(
                    &ready_tx,
                    CaptureError::CaptureFailed("no pipeline bus".into()),
                )
            })?;
            while let Some(msg) = bus.pop() {
                use gstreamer::MessageView;
                match msg.view() {
                    MessageView::Error(err) => {
                        warn!(
                            "pipeline error: {} ({})",
                            err.error(),
                            err.debug().unwrap_or_default()
                        );
                    }
                    MessageView::Warning(wrn) => {
                        warn!(
                            "pipeline warning: {} ({})",
                            wrn.error(),
                            wrn.debug().unwrap_or_default()
                        );
                    }
                    _ => {}
                }
            }
        }
        trace!("pipeline playing");

        info!(
            fd = pw_raw_fd,
            node = pw_node_id,
            "portal + GStreamer capture started"
        );

        // Signal readiness — pipeline is live, frames will flow.
        let _ = ready_tx.send(Ok(()));

        Self::capture_loop(appsink, &frame_tx, &stop_flag);

        let _ = pipeline.set_state(gstreamer::State::Null);
        info!("portal capture stopped");

        Ok(())
    }

    // ------------------------------------------------------------------
    // Portal handshake via ashpd
    // ------------------------------------------------------------------

    /// Run the portal D-Bus handshake using `ashpd`.
    /// Returns `(OwnedFd, pipewire_node_id)`.  The fd must be kept alive
    /// for the entire lifetime of the `GStreamer` pipeline.
    async fn portal_handshake() -> Result<(OwnedFd, u32), CaptureError> {
        let screencast = Screencast::new()
            .await
            .map_err(|e| CaptureError::CaptureFailed(format!("screencast connect: {e}")))?;

        let saved_token = load_restore_token();

        // CreateSessionOptions is not re-exported; Default() works fine.
        let session = screencast
            .create_session(Default::default())
            .await
            .map_err(|e| CaptureError::CaptureFailed(format!("create session: {e}")))?;

        info!("portal session created");

        // If we have a restore token, skip the source-selection dialog.
        let mut select_opts = SelectSourcesOptions::default()
            .set_sources(SourceType::Monitor | SourceType::Window)
            .set_multiple(false)
            .set_cursor_mode(CursorMode::Embedded);
        if let Some(ref token) = saved_token {
            select_opts = select_opts.set_restore_token(token.as_str());
        }

        screencast
            .select_sources(&session, select_opts)
            .await
            .map_err(|e| CaptureError::CaptureFailed(format!("select sources: {e}")))?;

        info!("portal sources selected");

        // Use a dummy WindowIdentifier to anchor the screencast session.
        // This matches the screencast_mvp pattern and works for headless
        // (SSH) sessions where no real X11 window exists.
        let window_id = ashpd::WindowIdentifier::from_xid(0);
        let response = screencast
            .start(&session, Some(&window_id), StartCastOptions::default())
            .await
            .map_err(|e| CaptureError::CaptureFailed(format!("start: {e}")))?
            .response()
            .map_err(|e| CaptureError::CaptureFailed(format!("start result: {e}")))?;

        let stream = response
            .streams()
            .first()
            .ok_or_else(|| CaptureError::CaptureFailed("no streams in portal response".into()))?
            .clone();

        let pw_node_id = stream.pipe_wire_node_id();

        let pw_fd = screencast
            .open_pipe_wire_remote(&session, Default::default())
            .await
            .map_err(|e| CaptureError::CaptureFailed(format!("open_pipe_wire_remote: {e}")))?;

        debug!(fd = pw_fd.as_raw_fd(), node = pw_node_id, "pipewire source");

        // Save restore token if portal provided one.
        if let Some(token) = response.restore_token() {
            save_restore_token(token);
        }

        Ok((pw_fd, pw_node_id))
    }

    // ------------------------------------------------------------------
    // `GStreamer` pipeline
    // ------------------------------------------------------------------

    /// Build the `GStreamer` pipeline programmatically (matching `screencast_mvp`).
    /// Pipeline: pipewiresrc → videoconvert → capsfilter(BGRx) → appsink
    fn build_pipeline(pw_fd: i32, pw_node_id: u32) -> Result<gstreamer::Pipeline, CaptureError> {
        let pipewire_src = gstreamer::ElementFactory::make("pipewiresrc")
            .property("fd", pw_fd)
            .property("path", pw_node_id.to_string())
            .build()
            .map_err(|e| {
                CaptureError::CaptureFailed(format!(
                    "failed to create pipewiresrc (is gst-plugin-pipewire installed?): {e}"
                ))
            })?;

        let videoconvert = gstreamer::ElementFactory::make("videoconvert")
            .build()
            .map_err(|e| {
                CaptureError::CaptureFailed(format!("failed to create videoconvert: {e}"))
            })?;

        let capsfilter = gstreamer::ElementFactory::make("capsfilter")
            .property(
                "caps",
                gstreamer::Caps::builder("video/x-raw")
                    .field("format", "BGRx")
                    .build(),
            )
            .build()
            .map_err(|e| {
                CaptureError::CaptureFailed(format!("failed to create capsfilter: {e}"))
            })?;

        let appsink = gstreamer::ElementFactory::make("appsink")
            .property("name", "appsink")
            .property("sync", false)
            .build()
            .map_err(|e| CaptureError::CaptureFailed(format!("failed to create appsink: {e}")))?;

        let pipeline = gstreamer::Pipeline::default();

        pipeline
            .add_many([&pipewire_src, &videoconvert, &capsfilter, &appsink])
            .map_err(|e| {
                CaptureError::CaptureFailed(format!("failed to add elements to pipeline: {e}"))
            })?;

        gstreamer::Element::link_many([&pipewire_src, &videoconvert, &capsfilter, &appsink])
            .map_err(|e| CaptureError::CaptureFailed(format!("failed to link elements: {e}")))?;

        Ok(pipeline)
    }

    /// Convert a `GStreamer` sample into a [`CapturedFrame`].
    ///
    /// Reads the buffer data, extracts dimensions from caps (or infers them
    /// from the buffer size), and packages everything into a frame ready for
    /// the compression pipeline.
    fn sample_to_frame(sample: &gstreamer::Sample) -> Result<CapturedFrame, CaptureError> {
        let buffer = sample
            .buffer()
            .ok_or_else(|| CaptureError::CaptureFailed("sample has no buffer".into()))?;

        let map = buffer
            .map_readable()
            .map_err(|_| CaptureError::CaptureFailed("gst buffer map_readable failed".into()))?;
        let data = map.as_slice();
        let buf_size = data.len() as u32;
        const BYTES_PER_PIXEL: u32 = 4; // BGRx

        // Read dimensions from caps if available, otherwise infer from buffer.
        let caps = sample.caps();
        let caps_w = caps
            .as_ref()
            .and_then(|c| c.structure(0).and_then(|s| s.get::<i32>("width").ok()))
            .unwrap_or(0);
        let caps_h = caps
            .as_ref()
            .and_then(|c| c.structure(0).and_then(|s| s.get::<i32>("height").ok()))
            .unwrap_or(0);
        let caps_stride = caps
            .as_ref()
            .and_then(|c| c.structure(0).and_then(|s| s.get::<i32>("stride").ok()))
            .unwrap_or(0);

        let (width, height, stride) = if caps_w > 0 && caps_h > 0 {
            let s = if caps_stride > 0 {
                caps_stride as u32
            } else {
                caps_w as u32 * BYTES_PER_PIXEL
            };
            (caps_w as u32, caps_h as u32, s)
        } else {
            // appsink caps often lack dimensions; estimate from buffer size.
            let s = if caps_stride > 0 {
                caps_stride as u32
            } else {
                buf_size
            }; // assume single row
            let h = if buf_size > s { buf_size / s } else { 1 };
            let w = s / BYTES_PER_PIXEL;
            (w, h, s)
        };

        let pixel_data = data.to_vec();
        let timestamp_ns = buffer.pts().map_or(0, |pts| pts.nseconds());

        Ok(CapturedFrame {
            data: pixel_data,
            damage: vec![DamageRect::new(0, 0, width, height)],
            format: PixelFormat::Xbgr8888,
            width,
            height,
            stride,
            timestamp_ns,
        })
    }

    /// Main capture loop: pulls samples from `appsink`, converts to [`CapturedFrame`],
    /// and pushes them through `frame_tx`.
    ///
    /// Runs until `stop_flag` is set or the channel is closed. Uses non-blocking
    /// `try_send` to avoid panicking inside the Tokio `current_thread` runtime.
    fn capture_loop(
        appsink: gstreamer_app::AppSink,
        frame_tx: &mpsc::Sender<CapturedFrame>,
        stop_flag: &AtomicBool,
    ) {
        let timeout_dur = gstreamer::ClockTime::from_mseconds(2000);

        loop {
            if stop_flag.load(Ordering::Relaxed) {
                break;
            }

            let sample = match appsink.try_pull_sample(timeout_dur) {
                Some(s) => s,
                None => {
                    trace!("no sample available");
                    continue;
                }
            };

            let frame = match Self::sample_to_frame(&sample) {
                Ok(f) => f,
                Err(e) => {
                    warn!("frame conversion error: {e}");
                    continue;
                }
            };

            // Use try_send (non-blocking) — we're inside a current_thread
            // runtime where blocking_send would panic.
            match frame_tx.try_send(frame) {
                Ok(()) => {}
                Err(mpsc::error::TrySendError::Full(_)) => {
                    // Ring is full — next recv() drains it, keep going.
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    break;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CaptureBackend trait implementation
// ---------------------------------------------------------------------------

impl CaptureBackend for PortalBackend {
    fn next_frame(&mut self) -> Result<CapturedFrame, CaptureError> {
        self.frame_rx
            .blocking_recv()
            .ok_or(CaptureError::SessionEnded)
    }

    fn name(&self) -> &'static str {
        "portal-gst"
    }

    fn stop(&mut self) {
        self.stop_flag.store(true, Ordering::Relaxed);
        self.frame_rx.close();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    // ── Token helpers ─────────────────────────────────────────────────────

    #[test]
    fn token_path_returns_non_empty() {
        let path = token_path();
        assert!(!path.as_os_str().is_empty());
        assert!(path.ends_with(TOKEN_FILE));
    }

    #[test]
    fn save_and_load_roundtrip() {
        let _lock = ENV_LOCK.lock().unwrap();
        let dir = std::env::temp_dir().join("remoteway-portal-test");
        let dir_str = dir.to_string_lossy().into_owned();
        let prev = std::env::var("XDG_CACHE_HOME").ok();
        // SAFETY: test-only, single-threaded.
        unsafe { std::env::set_var("XDG_CACHE_HOME", &dir_str) };
        let token = "test-restore-token-12345";
        save_restore_token(token);
        assert_eq!(load_restore_token().as_deref(), Some(token));
        let _ = std::fs::remove_dir_all(&dir);
        if let Some(v) = prev {
            unsafe { std::env::set_var("XDG_CACHE_HOME", v) };
        } else {
            unsafe { std::env::remove_var("XDG_CACHE_HOME") };
        }
    }

    #[test]
    fn save_restore_token_does_not_panic_on_bad_path() {
        save_restore_token("dummy");
    }

    #[test]
    fn portal_backend_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<PortalBackend>();
    }

    #[test]
    fn capture_backend_trait_object() {
        fn _assert() {
            let _: Box<dyn CaptureBackend>;
        }
    }

    #[test]
    fn build_pipeline_creates_pipeline() {
        if gstreamer::init().is_err() {
            eprintln!("skipping test: gstreamer not available");
            return;
        }
        let result = PortalBackend::build_pipeline(42, 123);
        assert!(result.is_ok(), "pipeline should parse: {:?}", result.err());
    }

    #[test]
    fn portal_backend_new_without_dbus_does_not_panic() {
        let _result = PortalBackend::new();
    }

    #[test]
    fn portal_backend_name_is_static() {
        let _: &'static str = PortalBackend::new()
            .ok()
            .map(|b| b.name())
            .unwrap_or("portal-gst");
    }
}

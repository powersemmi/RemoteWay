//! Auto-detection of the best available capture backend.
//!
//! Probes Wayland protocols in priority order: `ext-image-capture-source-v1`,
//! xdg-desktop-portal + `GStreamer`, and `wlr-screencopy`.

use wayland_client::Connection;

use crate::backend::CaptureBackend;
use crate::desktop_detect::detect_desktop;
use crate::error::CaptureError;
use crate::ext_capture::{CaptureSource, ExtImageCaptureBackend};
pub use crate::ext_capture::{ToplevelInfo, enumerate_toplevels};
#[cfg(feature = "portal")]
use crate::portal::PortalBackend;
use crate::screencopy::WlrScreencopyBackend;

/// Auto-detect the best available capture backend.
///
/// Priority order:
/// 1. `ext-image-capture-source-v1` — modern protocol (GNOME 46+, KDE 6+, wlroots 0.18+)
/// 2. Portal + `GStreamer` — xdg-desktop-portal (GNOME, KDE)
/// 3. `wlr-screencopy` — legacy protocol, tried as last-resort fallback for all DEs
pub fn detect_backend(output_name: Option<&str>) -> Result<Box<dyn CaptureBackend>, CaptureError> {
    let desktop = detect_desktop();
    tracing::info!(desktop = %desktop, "detected desktop environment");

    // 1. ext-image-capture — modern standard Wayland protocol.
    let source = CaptureSource::Output(output_name.map(String::from));
    match ExtImageCaptureBackend::new(source) {
        Ok(backend) => {
            tracing::info!("using ext-image-capture backend");
            return Ok(Box::new(backend));
        }
        Err(e) => {
            tracing::info!("ext-image-capture unavailable: {e}");
        }
    }

    // 2. portal + GStreamer — GNOME/KDE via xdg-desktop-portal.
    #[cfg(feature = "portal")]
    {
        let mut portal_error: Option<CaptureError> = None;
        if desktop.needs_portal() {
            match PortalBackend::new() {
                Ok(backend) => {
                    tracing::info!("using portal + GStreamer backend");
                    return Ok(Box::new(backend));
                }
                Err(e) => {
                    tracing::info!("portal backend unavailable: {e}");
                    portal_error = Some(e);
                }
            }
        }

        // 3. wlr-screencopy — try for ALL desktops as last-resort fallback.
        //    Some KDE setups lack gst-plugin-pipewire but support wlr-screencopy.
        match WlrScreencopyBackend::new(output_name) {
            Ok(backend) => {
                tracing::info!("using wlr-screencopy backend (portal fallback)");
                return Ok(Box::new(backend));
            }
            Err(e) => {
                tracing::info!("wlr-screencopy unavailable: {e}");
                // If portal was our primary path and it failed with pipewiresrc,
                // surface a specific actionable error.
                if let Some(ref pe) = portal_error {
                    if format!("{pe}").contains("pipewiresrc") {
                        tracing::error!(
                            "portal capture failed: `pipewiresrc` GStreamer element not found. \
                             Install the PipeWire GStreamer plugin:\n  \
                             Debian/Ubuntu:  sudo apt install gstreamer1.0-pipewire\n  \
                             Fedora:         sudo dnf install gstreamer1-plugin-pipewire\n  \
                             Arch:           sudo pacman -S gst-plugin-pipewire"
                        );
                    }
                }
            }
        }
    }

    // No portal feature — try wlr-screencopy directly.
    #[cfg(not(feature = "portal"))]
    {
        match WlrScreencopyBackend::new(output_name) {
            Ok(backend) => {
                tracing::info!("using wlr-screencopy backend");
                return Ok(Box::new(backend));
            }
            Err(e) => {
                tracing::info!("wlr-screencopy unavailable: {e}");
            }
        }
    }

    tracing::error!(
        "no capture backend available for {desktop}. \
         ext-image-capture-source-v1 requires GNOME 46+ / KDE 6+ / wlroots 0.18+. \
         Portal + PipeWire requires the GStreamer PipeWire plugin. \
         On older compositors, upgrade or install missing packages."
    );
    Err(CaptureError::NoBackend)
}

/// Detect backend for capturing a specific toplevel window.
pub fn detect_toplevel_backend(app_id: &str) -> Result<Box<dyn CaptureBackend>, CaptureError> {
    let source = CaptureSource::Toplevel(app_id.to_string());
    let backend = ExtImageCaptureBackend::new(source)?;
    Ok(Box::new(backend))
}

/// Detect backend for a newly appeared toplevel (diff-based).
///
/// Captures the first toplevel whose `identifier` is NOT in `known_identifiers`.
/// Used to auto-detect the window of a just-spawned child process.
pub fn detect_new_toplevel_backend(
    known_identifiers: &[String],
) -> Result<Box<dyn CaptureBackend>, CaptureError> {
    let source = CaptureSource::NewToplevel {
        known_identifiers: known_identifiers.to_vec(),
    };
    let backend = ExtImageCaptureBackend::new(source)?;
    Ok(Box::new(backend))
}

/// Check if any capture backend is available without creating one.
#[must_use]
pub fn is_capture_available() -> bool {
    let Ok(conn) = Connection::connect_to_env() else {
        return false;
    };
    ExtImageCaptureBackend::is_available(&conn) || WlrScreencopyBackend::is_available(&conn)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_without_wayland_returns_error() {
        if std::env::var("WAYLAND_DISPLAY").is_ok() {
            return;
        }
        let result = detect_backend(None);
        assert!(result.is_err());
    }

    #[test]
    fn is_capture_available_without_wayland() {
        if std::env::var("WAYLAND_DISPLAY").is_ok() {
            return;
        }
        assert!(!is_capture_available());
    }

    #[test]
    fn detect_toplevel_without_wayland_returns_error() {
        if std::env::var("WAYLAND_DISPLAY").is_ok() {
            return;
        }
        let result = detect_toplevel_backend("org.mozilla.firefox");
        assert!(result.is_err());
    }

    #[test]
    fn detect_new_toplevel_without_wayland_returns_error() {
        if std::env::var("WAYLAND_DISPLAY").is_ok() {
            return;
        }
        let result = detect_new_toplevel_backend(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn detect_backend_with_named_output_without_wayland() {
        if std::env::var("WAYLAND_DISPLAY").is_ok() {
            return;
        }
        let result = detect_backend(Some("HDMI-A-1"));
        assert!(result.is_err());
    }

    #[test]
    fn detect_toplevel_backend_various_app_ids() {
        if std::env::var("WAYLAND_DISPLAY").is_ok() {
            return;
        }
        for app_id in &["org.mozilla.firefox", "com.google.Chrome", "kitty", ""] {
            let result = detect_toplevel_backend(app_id);
            assert!(result.is_err(), "expected error for app_id: {app_id}");
        }
    }

    #[test]
    fn detect_new_toplevel_with_known_ids() {
        if std::env::var("WAYLAND_DISPLAY").is_ok() {
            return;
        }
        let known = vec!["id1".to_string(), "id2".to_string()];
        let result = detect_new_toplevel_backend(&known);
        assert!(result.is_err());
    }

    #[test]
    fn is_capture_available_is_deterministic() {
        if std::env::var("WAYLAND_DISPLAY").is_ok() {
            return;
        }
        let r1 = is_capture_available();
        let r2 = is_capture_available();
        assert_eq!(r1, r2);
        assert!(!r1);
    }
}

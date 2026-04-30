use thiserror::Error;

#[derive(Debug, Error)]
pub enum CaptureError {
    #[error("wayland connection error: {0}")]
    WaylandConnect(#[from] wayland_client::ConnectError),
    #[error("wayland dispatch error: {0}")]
    WaylandDispatch(#[from] wayland_client::DispatchError),
    #[error("no suitable capture protocol available")]
    NoBackend,
    #[error("no outputs found")]
    NoOutputs,
    #[error("shm pool error: {0}")]
    ShmPool(String),
    #[error("buffer pool exhausted")]
    BufferPoolExhausted,
    #[error("capture session ended by compositor")]
    SessionEnded,
    #[error("output {0} not found")]
    OutputNotFound(String),
    #[error("unsupported pixel format: {0}")]
    UnsupportedFormat(u32),
    #[error("thread spawn failed: {0}")]
    ThreadSpawn(#[from] remoteway_core::thread_config::ThreadConfigError),
    #[error("capture failed: {0}")]
    CaptureFailed(String),
    #[error("protocol error: {0}")]
    Protocol(String),
}

impl CaptureError {
    /// Whether this error is transient and the operation may succeed if retried.
    ///
    /// `NoBackend` and `WaylandConnect` are permanent — the required protocol
    /// or compositor won't appear on retry. `CaptureFailed` may be transient
    /// (e.g. a toplevel not yet mapped).
    pub fn is_retriable(&self) -> bool {
        matches!(self, CaptureError::CaptureFailed(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_all_variants() {
        let errors: Vec<CaptureError> = vec![
            CaptureError::NoBackend,
            CaptureError::NoOutputs,
            CaptureError::ShmPool("test".into()),
            CaptureError::BufferPoolExhausted,
            CaptureError::SessionEnded,
            CaptureError::OutputNotFound("HDMI-A-1".into()),
            CaptureError::UnsupportedFormat(42),
            CaptureError::CaptureFailed("timeout".into()),
        ];
        for e in &errors {
            let s = e.to_string();
            assert!(!s.is_empty(), "empty display for {:?}", e);
        }
    }

    #[test]
    fn error_is_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<CaptureError>();
        assert_sync::<CaptureError>();
    }

    #[test]
    fn is_retriable_capture_failed() {
        assert!(CaptureError::CaptureFailed("transient".into()).is_retriable());
    }

    #[test]
    fn is_retriable_permanent_errors() {
        assert!(!CaptureError::NoBackend.is_retriable());
        assert!(!CaptureError::NoOutputs.is_retriable());
        assert!(!CaptureError::ShmPool("err".into()).is_retriable());
        assert!(!CaptureError::BufferPoolExhausted.is_retriable());
        assert!(!CaptureError::SessionEnded.is_retriable());
        assert!(!CaptureError::OutputNotFound("x".into()).is_retriable());
        assert!(!CaptureError::UnsupportedFormat(0).is_retriable());
        assert!(!CaptureError::Protocol("x".into()).is_retriable());
    }

    #[test]
    fn error_debug_output() {
        let e = CaptureError::OutputNotFound("DP-1".into());
        let dbg = format!("{:?}", e);
        assert!(dbg.contains("OutputNotFound"));
        assert!(dbg.contains("DP-1"));
    }

    #[test]
    fn error_display_protocol() {
        let e = CaptureError::Protocol("protocol mismatch".into());
        assert!(e.to_string().contains("protocol mismatch"));
    }
}

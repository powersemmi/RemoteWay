use thiserror::Error;

/// Errors that can occur during display / window management.
#[derive(Debug, Error)]
pub enum DisplayError {
    /// Failed to connect to the Wayland compositor.
    #[error("wayland connection error: {0}")]
    WaylandConnect(#[from] wayland_client::ConnectError),
    /// A Wayland dispatch I/O error occurred.
    #[error("wayland dispatch error: {0}")]
    WaylandDispatch(#[from] wayland_client::DispatchError),
    /// The compositor does not expose `wl_compositor`.
    #[error("no wl_compositor global found")]
    NoCompositor,
    /// The compositor does not expose `wl_shm`.
    #[error("no wl_shm global found")]
    NoShm,
    /// The compositor does not expose `xdg_wm_base`.
    #[error("no xdg_wm_base global found")]
    NoXdgWmBase,
    /// No `wl_seat` global was found.
    #[error("no wl_seat global found")]
    NoSeat,
    /// Failed to create or resize a shared-memory buffer.
    #[error("shm buffer error: {0}")]
    ShmBuffer(String),
    /// The Wayland surface has not received a configure event yet.
    #[error("surface not configured")]
    NotConfigured,
    /// A surface with the given ID was not found.
    #[error("surface {0} not found")]
    SurfaceNotFound(u16),
    /// Failed to spawn an internal helper thread.
    #[error("thread spawn failed: {0}")]
    ThreadSpawn(#[from] remoteway_core::thread_config::ThreadConfigError),
    /// The display session was cleanly ended.
    #[error("display session ended")]
    SessionEnded,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_all_variants() {
        let errors: Vec<DisplayError> = vec![
            DisplayError::NoCompositor,
            DisplayError::NoShm,
            DisplayError::NoXdgWmBase,
            DisplayError::NoSeat,
            DisplayError::ShmBuffer("test".into()),
            DisplayError::NotConfigured,
            DisplayError::SurfaceNotFound(42),
            DisplayError::SessionEnded,
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
        assert_send::<DisplayError>();
        assert_sync::<DisplayError>();
    }
}

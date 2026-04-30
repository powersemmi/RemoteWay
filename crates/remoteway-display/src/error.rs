use thiserror::Error;

#[derive(Debug, Error)]
pub enum DisplayError {
    #[error("wayland connection error: {0}")]
    WaylandConnect(#[from] wayland_client::ConnectError),
    #[error("wayland dispatch error: {0}")]
    WaylandDispatch(#[from] wayland_client::DispatchError),
    #[error("no wl_compositor global found")]
    NoCompositor,
    #[error("no wl_shm global found")]
    NoShm,
    #[error("no xdg_wm_base global found")]
    NoXdgWmBase,
    #[error("no wl_seat global found")]
    NoSeat,
    #[error("shm buffer error: {0}")]
    ShmBuffer(String),
    #[error("surface not configured")]
    NotConfigured,
    #[error("surface {0} not found")]
    SurfaceNotFound(u16),
    #[error("thread spawn failed: {0}")]
    ThreadSpawn(#[from] remoteway_core::thread_config::ThreadConfigError),
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

use thiserror::Error;

#[derive(Debug, Error)]
pub enum InputError {
    #[error("wayland connection error: {0}")]
    WaylandConnect(#[from] wayland_client::ConnectError),
    #[error("wayland dispatch error: {0}")]
    WaylandDispatch(#[from] wayland_client::DispatchError),
    #[error("virtual pointer manager not available")]
    NoVirtualPointer,
    #[error("virtual keyboard manager not available")]
    NoVirtualKeyboard,
    #[error("no wl_seat available")]
    NoSeat,
    #[error("keymap creation failed: {0}")]
    Keymap(String),
    #[error("unknown input event kind: {0}")]
    UnknownInputKind(u8),
    #[error("thread spawn failed: {0}")]
    ThreadSpawn(#[from] remoteway_core::thread_config::ThreadConfigError),
    #[error("inject failed: {0}")]
    InjectFailed(String),
    #[error("input session ended")]
    SessionEnded,
    #[error("protocol error: {0}")]
    Protocol(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_all_variants() {
        let errors: Vec<InputError> = vec![
            InputError::NoVirtualPointer,
            InputError::NoVirtualKeyboard,
            InputError::NoSeat,
            InputError::Keymap("test".into()),
            InputError::UnknownInputKind(99),
            InputError::InjectFailed("timeout".into()),
            InputError::SessionEnded,
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
        assert_send::<InputError>();
        assert_sync::<InputError>();
    }
}

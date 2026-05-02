//! Error types for input event capture and injection operations.

use thiserror::Error;

/// Errors that can occur during input event capture and injection.
#[derive(Debug, Error)]
pub enum InputError {
    /// Failed to connect to the Wayland compositor.
    #[error("wayland connection error: {0}")]
    WaylandConnect(#[from] wayland_client::ConnectError),
    /// A Wayland dispatch I/O error occurred.
    #[error("wayland dispatch error: {0}")]
    WaylandDispatch(#[from] wayland_client::DispatchError),
    /// The compositor does not expose `zwp_virtual_pointer_manager_v1`.
    #[error("virtual pointer manager not available")]
    NoVirtualPointer,
    /// The compositor does not expose `zwp_virtual_keyboard_manager_v1`.
    #[error("virtual keyboard manager not available")]
    NoVirtualKeyboard,
    /// No `wl_seat` global was found.
    #[error("no wl_seat available")]
    NoSeat,
    /// Failed to create an xkb keymap from the compositor's keymap string.
    #[error("keymap creation failed: {0}")]
    Keymap(String),
    /// Received an input event with an unrecognised kind byte.
    #[error("unknown input event kind: {0}")]
    UnknownInputKind(u8),
    /// Failed to spawn an internal helper thread.
    #[error("thread spawn failed: {0}")]
    ThreadSpawn(#[from] remoteway_core::thread_config::ThreadConfigError),
    /// Injecting an input event into the compositor failed.
    #[error("inject failed: {0}")]
    InjectFailed(String),
    /// The input session was cleanly ended.
    #[error("input session ended")]
    SessionEnded,
    /// A Wayland protocol error was received.
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

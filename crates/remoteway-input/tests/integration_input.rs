//! Integration tests for remoteway-input crate.
//!
//! Tests that don't require a real Wayland compositor.
//! Mock compositor tests are in mock_virtual_input.rs.

use remoteway_proto::input::{
    InputEvent, InputKind, KeyEvent, PointerAxis, PointerButton, PointerMotion,
};

use remoteway_input::error::InputError;

// --- Error type tests ---

#[test]
fn error_variants_display() {
    let errors: Vec<InputError> = vec![
        InputError::NoVirtualPointer,
        InputError::NoVirtualKeyboard,
        InputError::NoSeat,
        InputError::Keymap("test failure".into()),
        InputError::UnknownInputKind(42),
        InputError::InjectFailed("protocol error".into()),
        InputError::SessionEnded,
    ];
    for e in &errors {
        let s = e.to_string();
        assert!(!s.is_empty());
    }
}

#[test]
fn error_from_thread_config_error() {
    let err: InputError =
        remoteway_core::thread_config::ThreadConfigError::Spawn(std::io::Error::other("test"))
            .into();
    assert!(err.to_string().contains("thread spawn"));
}

// --- Keymap tests ---

#[test]
fn default_keymap_contains_expected_sections() {
    let km = remoteway_input::keymap::DEFAULT_KEYMAP;
    assert!(km.contains("xkb_keymap"));
    assert!(km.contains("xkb_keycodes"));
    assert!(km.contains("xkb_types"));
    assert!(km.contains("xkb_compatibility"));
    assert!(km.contains("xkb_symbols"));
}

#[test]
fn create_keymap_fd_round_trip() {
    use std::io::{Read, Seek, SeekFrom};

    let keymap_str = "test xkb keymap content";
    let (fd, size) = remoteway_input::keymap::create_keymap_fd(keymap_str).unwrap();
    assert_eq!(size, keymap_str.len() as u32 + 1);

    let mut file = std::fs::File::from(fd);
    file.seek(SeekFrom::Start(0)).unwrap();
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).unwrap();
    assert_eq!(buf.len(), size as usize);
    assert_eq!(&buf[..buf.len() - 1], keymap_str.as_bytes());
    assert_eq!(*buf.last().unwrap(), 0u8); // NUL terminator
}

#[test]
fn create_keymap_fd_with_default_keymap() {
    let (fd, size) =
        remoteway_input::keymap::create_keymap_fd(remoteway_input::keymap::DEFAULT_KEYMAP).unwrap();
    assert!(size > 100);
    drop(fd);
}

// --- InputEvent serialization round-trip ---

#[test]
fn all_event_kinds_round_trip() {
    let events = vec![
        (
            InputEvent::pointer_motion(PointerMotion {
                surface_id: 1,
                _pad: 0,
                x: 100.5,
                y: 200.3,
            }),
            InputKind::PointerMotion,
        ),
        (
            InputEvent::pointer_button(PointerButton {
                button: 0x110,
                state: 1,
            }),
            InputKind::PointerButton,
        ),
        (
            InputEvent::pointer_axis(PointerAxis {
                axis: 0,
                _pad: [0; 3],
                value: -15.0,
            }),
            InputKind::PointerAxis,
        ),
        (
            InputEvent::key(KeyEvent { key: 30, state: 1 }),
            InputKind::Key,
        ),
    ];
    for (ev, expected_kind) in &events {
        assert_eq!(ev.kind().unwrap(), *expected_kind);
        assert_eq!(std::mem::size_of_val(ev), 16);
    }
}

// --- Config defaults ---

#[test]
fn inject_config_default_values() {
    let cfg = remoteway_input::inject_thread::InputInjectConfig::default();
    assert_eq!(cfg.core_id, 0);
    assert_eq!(cfg.sched_priority, 99);
    assert_eq!(cfg.ring_capacity, 256);
}

#[test]
fn capture_config_default_values() {
    let cfg = remoteway_input::capture_thread::InputCaptureConfig::default();
    assert_eq!(cfg.core_id, 0);
    assert_eq!(cfg.sched_priority, 99);
}

// --- Graceful failure without compositor ---

#[test]
fn virtual_input_new_fails_without_wayland() {
    // Ensure no WAYLAND_DISPLAY is set.
    let saved = std::env::var("WAYLAND_DISPLAY").ok();
    // SAFETY: test environment, single-threaded access to env var.
    unsafe { std::env::remove_var("WAYLAND_DISPLAY") };

    let result = remoteway_input::inject::VirtualInput::new();
    assert!(result.is_err());

    // Restore env.
    if let Some(val) = saved {
        // SAFETY: restoring previously saved value.
        unsafe { std::env::set_var("WAYLAND_DISPLAY", val) };
    }
}

#[test]
fn input_capture_new_fails_without_wayland() {
    let saved = std::env::var("WAYLAND_DISPLAY").ok();
    // SAFETY: test environment, single-threaded access to env var.
    unsafe { std::env::remove_var("WAYLAND_DISPLAY") };

    let result = remoteway_input::capture::InputCapture::new();
    assert!(result.is_err());

    if let Some(val) = saved {
        // SAFETY: restoring previously saved value.
        unsafe { std::env::set_var("WAYLAND_DISPLAY", val) };
    }
}

// --- Additional error coverage ---

#[test]
fn error_protocol_variant() {
    let err = InputError::Protocol("unexpected message".to_string());
    let display = err.to_string();
    assert!(display.contains("protocol error"));
    assert!(display.contains("unexpected message"));
}

#[test]
fn error_session_ended_variant() {
    let err = InputError::SessionEnded;
    assert!(err.to_string().contains("input session ended"));
}

#[test]
fn error_inject_failed_variant() {
    let err = InputError::InjectFailed("wayland read: broken pipe".to_string());
    let display = err.to_string();
    assert!(display.contains("inject failed"));
    assert!(display.contains("broken pipe"));
}

// --- InputEvent serialization edge cases ---

#[test]
fn input_event_is_exactly_16_bytes() {
    assert_eq!(std::mem::size_of::<InputEvent>(), 16);
}

#[test]
fn input_event_copy_semantics() {
    let ev = InputEvent::key(KeyEvent { key: 30, state: 1 });
    let ev_copy = ev;
    // Both should be valid (Copy type).
    assert_eq!(ev.kind, ev_copy.kind);
    assert_eq!(ev.payload, ev_copy.payload);
}

#[test]
fn pointer_motion_all_fields_round_trip() {
    use zerocopy::FromBytes;

    let motion = PointerMotion {
        surface_id: 12345,
        _pad: 0,
        x: -1.5,
        y: 999.999,
    };
    let ev = InputEvent::pointer_motion(motion);
    let decoded =
        PointerMotion::ref_from_bytes(&ev.payload[..std::mem::size_of::<PointerMotion>()]).unwrap();
    assert_eq!({ decoded.surface_id }, 12345);
    assert_eq!({ decoded.x }, -1.5);
    assert!(({ decoded.y } - 999.999).abs() < 0.01);
}

// --- Config edge cases ---

#[test]
fn inject_config_zero_capacity() {
    // rtrb::RingBuffer::new(0) should still work (creates capacity 1 internally).
    let cfg = remoteway_input::inject_thread::InputInjectConfig {
        core_id: 0,
        sched_priority: 0,
        ring_capacity: 0,
    };
    assert_eq!(cfg.ring_capacity, 0);
}

#[test]
fn capture_config_max_values() {
    let cfg = remoteway_input::capture_thread::InputCaptureConfig {
        core_id: usize::MAX,
        sched_priority: u8::MAX,
    };
    assert_eq!(cfg.core_id, usize::MAX);
    assert_eq!(cfg.sched_priority, u8::MAX);
}

// --- Keymap edge cases ---

#[test]
fn create_keymap_fd_unicode_content() {
    let content = "xkb_keymap { /* \u{00e9}\u{00e8}\u{00ea} */ };";
    let result = remoteway_input::keymap::create_keymap_fd(content);
    assert!(result.is_ok());
    let (_, size) = result.unwrap();
    assert_eq!(size, content.len() as u32 + 1);
}

// --- Thread spawn patterns ---

#[test]
fn inject_thread_spawn_and_immediate_stop() {
    let saved = std::env::var("WAYLAND_DISPLAY").ok();
    // SAFETY: test environment.
    unsafe { std::env::remove_var("WAYLAND_DISPLAY") };

    let config = remoteway_input::inject_thread::InputInjectConfig {
        core_id: 0,
        sched_priority: 0,
        ring_capacity: 4,
    };
    let result = remoteway_input::inject_thread::InputInjectThread::spawn(config);
    if let Ok(mut thread) = result {
        thread.stop();
    }

    if let Some(val) = saved {
        // SAFETY: restoring.
        unsafe { std::env::set_var("WAYLAND_DISPLAY", val) };
    }
}

#[test]
fn capture_thread_spawn_and_immediate_stop() {
    let saved = std::env::var("WAYLAND_DISPLAY").ok();
    // SAFETY: test environment.
    unsafe { std::env::remove_var("WAYLAND_DISPLAY") };

    let sender: remoteway_input::capture_thread::EventSender = Box::new(|_| true);
    let config = remoteway_input::capture_thread::InputCaptureConfig {
        core_id: 0,
        sched_priority: 0,
    };
    let result = remoteway_input::capture_thread::InputCaptureThread::spawn(config, sender);
    if let Ok(mut thread) = result {
        thread.stop();
    }

    if let Some(val) = saved {
        // SAFETY: restoring.
        unsafe { std::env::set_var("WAYLAND_DISPLAY", val) };
    }
}

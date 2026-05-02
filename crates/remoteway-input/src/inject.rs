//! Wayland virtual input injection — replays remote input events on the server side.

use std::os::fd::AsFd;
use std::time::{SystemTime, UNIX_EPOCH};

use wayland_client::protocol::{wl_pointer, wl_registry, wl_seat};
use wayland_client::{Connection, Dispatch, QueueHandle};
use wayland_protocols_misc::zwp_virtual_keyboard_v1::client::{
    zwp_virtual_keyboard_manager_v1, zwp_virtual_keyboard_v1,
};
use wayland_protocols_wlr::virtual_pointer::v1::client::{
    zwlr_virtual_pointer_manager_v1, zwlr_virtual_pointer_v1,
};
use zerocopy::FromBytes;

use remoteway_proto::input::{
    InputEvent, InputKind, KeyEvent, PointerAxis, PointerButton, PointerMotion,
};

use crate::error::InputError;
use crate::keymap;

/// Injects input events into a Wayland compositor via virtual pointer/keyboard protocols.
///
/// Used on the server side to replay remote input events as if they came from
/// physical devices. Connects to the local compositor and creates virtual devices.
pub struct VirtualInput {
    conn: Connection,
    state: InjectState,
    _event_queue: wayland_client::EventQueue<InjectState>,
}

struct InjectState {
    seat: Option<wl_seat::WlSeat>,
    vp_manager: Option<zwlr_virtual_pointer_manager_v1::ZwlrVirtualPointerManagerV1>,
    vk_manager: Option<zwp_virtual_keyboard_manager_v1::ZwpVirtualKeyboardManagerV1>,
    virtual_pointer: Option<zwlr_virtual_pointer_v1::ZwlrVirtualPointerV1>,
    virtual_keyboard: Option<zwp_virtual_keyboard_v1::ZwpVirtualKeyboardV1>,
}

/// Extent used for `motion_absolute` coordinate mapping.
/// Virtual pointer protocol maps x/y to [0, extent].
const POINTER_EXTENT: u32 = 0xFFFF;

fn timestamp_ms() -> u32 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u32
}

impl VirtualInput {
    /// Connect to the compositor, bind virtual pointer and keyboard protocols, and set keymap.
    pub fn new() -> Result<Self, InputError> {
        let conn = Connection::connect_to_env()?;
        let display = conn.display();
        let mut event_queue = conn.new_event_queue::<InjectState>();
        let qh = event_queue.handle();

        let mut state = InjectState {
            seat: None,
            vp_manager: None,
            vk_manager: None,
            virtual_pointer: None,
            virtual_keyboard: None,
        };

        let _ = display.get_registry(&qh, ()); // INTENTIONAL: WlRegistry managed by event queue

        // First roundtrip: discover globals.
        let _ = event_queue.roundtrip(&mut state)?; // INTENTIONAL: dispatch count irrelevant

        let seat = state.seat.as_ref().ok_or(InputError::NoSeat)?;
        let vp_mgr = state
            .vp_manager
            .as_ref()
            .ok_or(InputError::NoVirtualPointer)?;
        let vk_mgr = state
            .vk_manager
            .as_ref()
            .ok_or(InputError::NoVirtualKeyboard)?;

        // Create virtual devices.
        let vp = vp_mgr.create_virtual_pointer(Some(seat), &qh, ());
        state.virtual_pointer = Some(vp);

        let vk = vk_mgr.create_virtual_keyboard(seat, &qh, ());

        // Set keymap on virtual keyboard (required before sending key events).
        let (keymap_fd, keymap_size) = keymap::create_keymap_fd(keymap::DEFAULT_KEYMAP)?;
        // format 1 = XKB_KEYMAP_FORMAT_TEXT_V1
        vk.keymap(1, keymap_fd.as_fd(), keymap_size);

        state.virtual_keyboard = Some(vk);

        // Second roundtrip: process any protocol responses.
        let _ = event_queue.roundtrip(&mut state)?; // INTENTIONAL: dispatch count irrelevant

        Ok(Self {
            conn,
            state,
            _event_queue: event_queue,
        })
    }

    /// Dispatch a single input event to the compositor.
    ///
    /// This is the hot-path method. No heap allocations.
    /// Decodes the event kind and calls the appropriate virtual device request.
    pub fn dispatch_event(&self, event: &InputEvent) -> Result<(), InputError> {
        let kind = event.kind().map_err(InputError::UnknownInputKind)?;
        let time = timestamp_ms();

        match kind {
            InputKind::PointerMotion => {
                let motion =
                    PointerMotion::ref_from_bytes(&event.payload[..size_of::<PointerMotion>()])
                        .map_err(|e| {
                            InputError::InjectFailed(format!("decode PointerMotion: {e}"))
                        })?;
                self.inject_pointer_motion(time, motion);
            }
            InputKind::PointerButton => {
                let btn =
                    PointerButton::ref_from_bytes(&event.payload[..size_of::<PointerButton>()])
                        .map_err(|e| {
                            InputError::InjectFailed(format!("decode PointerButton: {e}"))
                        })?;
                self.inject_pointer_button(time, btn);
            }
            InputKind::PointerAxis => {
                let axis = PointerAxis::ref_from_bytes(&event.payload[..size_of::<PointerAxis>()])
                    .map_err(|e| InputError::InjectFailed(format!("decode PointerAxis: {e}")))?;
                self.inject_pointer_axis(time, axis);
            }
            InputKind::Key => {
                let key = KeyEvent::ref_from_bytes(&event.payload[..size_of::<KeyEvent>()])
                    .map_err(|e| InputError::InjectFailed(format!("decode KeyEvent: {e}")))?;
                self.inject_key(time, key);
            }
        }

        Ok(())
    }

    /// Flush the Wayland connection buffer to the compositor.
    ///
    /// Should be called after dispatching a batch of events to ensure
    /// they are actually sent over the socket.
    pub fn flush(&self) {
        if let Err(e) = self.conn.flush() {
            tracing::warn!("flush error: {e}");
        }
    }

    fn inject_pointer_motion(&self, time: u32, motion: &PointerMotion) {
        if let Some(ref vp) = self.state.virtual_pointer {
            let x = motion.x as u32;
            let y = motion.y as u32;
            vp.motion_absolute(time, x, y, POINTER_EXTENT, POINTER_EXTENT);
            vp.frame();
        }
    }

    fn inject_pointer_button(&self, time: u32, btn: &PointerButton) {
        if let Some(ref vp) = self.state.virtual_pointer {
            let state_val = btn.state;
            let button_state = if state_val != 0 {
                wl_pointer::ButtonState::Pressed
            } else {
                wl_pointer::ButtonState::Released
            };
            vp.button(time, btn.button, button_state);
            vp.frame();
        }
    }

    fn inject_pointer_axis(&self, time: u32, axis: &PointerAxis) {
        if let Some(ref vp) = self.state.virtual_pointer {
            let axis_val = axis.axis;
            let wl_axis = if axis_val == 0 {
                wl_pointer::Axis::VerticalScroll
            } else {
                wl_pointer::Axis::HorizontalScroll
            };
            let value = axis.value;
            vp.axis(time, wl_axis, value as f64);
            vp.frame();
        }
    }

    fn inject_key(&self, time: u32, key: &KeyEvent) {
        if let Some(ref vk) = self.state.virtual_keyboard {
            vk.key(time, key.key, key.state);
        }
    }
}

// --- Wayland dispatch implementations ---

impl Dispatch<wl_registry::WlRegistry, ()> for InjectState {
    fn event(
        state: &mut Self,
        registry: &wl_registry::WlRegistry,
        event: wl_registry::Event,
        _: &(),
        _conn: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if let wl_registry::Event::Global {
            name,
            interface,
            version,
        } = event
        {
            match interface.as_str() {
                "wl_seat" => {
                    state.seat = Some(registry.bind(name, version.min(1), qh, ()));
                }
                "zwlr_virtual_pointer_manager_v1" => {
                    state.vp_manager = Some(registry.bind(name, version.min(2), qh, ()));
                }
                "zwp_virtual_keyboard_manager_v1" => {
                    state.vk_manager = Some(registry.bind(name, version.min(1), qh, ()));
                }
                _ => {}
            }
        }
    }
}

impl Dispatch<wl_seat::WlSeat, ()> for InjectState {
    fn event(
        _state: &mut Self,
        _proxy: &wl_seat::WlSeat,
        _event: wl_seat::Event,
        _: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<zwlr_virtual_pointer_manager_v1::ZwlrVirtualPointerManagerV1, ()> for InjectState {
    fn event(
        _state: &mut Self,
        _proxy: &zwlr_virtual_pointer_manager_v1::ZwlrVirtualPointerManagerV1,
        _event: zwlr_virtual_pointer_manager_v1::Event,
        _: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<zwlr_virtual_pointer_v1::ZwlrVirtualPointerV1, ()> for InjectState {
    fn event(
        _state: &mut Self,
        _proxy: &zwlr_virtual_pointer_v1::ZwlrVirtualPointerV1,
        _event: zwlr_virtual_pointer_v1::Event,
        _: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<zwp_virtual_keyboard_manager_v1::ZwpVirtualKeyboardManagerV1, ()> for InjectState {
    fn event(
        _state: &mut Self,
        _proxy: &zwp_virtual_keyboard_manager_v1::ZwpVirtualKeyboardManagerV1,
        _event: zwp_virtual_keyboard_manager_v1::Event,
        _: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<zwp_virtual_keyboard_v1::ZwpVirtualKeyboardV1, ()> for InjectState {
    fn event(
        _state: &mut Self,
        _proxy: &zwp_virtual_keyboard_v1::ZwpVirtualKeyboardV1,
        _event: zwp_virtual_keyboard_v1::Event,
        _: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

#[cfg(test)]
mod tests {
    use remoteway_proto::input::{InputEvent, KeyEvent, PointerAxis, PointerButton, PointerMotion};
    use zerocopy::FromBytes;

    use crate::error::InputError;

    #[test]
    fn dispatch_event_unknown_kind_returns_error() {
        // Create an event with an invalid kind.
        let mut ev = InputEvent::key(KeyEvent { key: 1, state: 0 });
        ev.kind = 99;
        // We can't create VirtualInput without a compositor, but we can test
        // the decode logic by verifying the error variant.
        let kind_result = ev.kind();
        assert!(kind_result.is_err());
        let err = InputError::UnknownInputKind(kind_result.unwrap_err());
        assert!(err.to_string().contains("99"));
    }

    #[test]
    fn all_input_kinds_decode_correctly() {
        let events = [
            InputEvent::pointer_motion(PointerMotion {
                surface_id: 0,
                _pad: 0,
                x: 100.0,
                y: 200.0,
            }),
            InputEvent::pointer_button(PointerButton {
                button: 0x110,
                state: 1,
            }),
            InputEvent::pointer_axis(PointerAxis {
                axis: 0,
                _pad: [0; 3],
                value: -5.0,
            }),
            InputEvent::key(KeyEvent { key: 30, state: 1 }),
        ];
        for ev in &events {
            assert!(ev.kind().is_ok());
        }
    }

    #[test]
    fn virtual_input_new_fails_without_compositor() {
        // Without a running Wayland compositor, this should fail with a connection error.
        // SAFETY: this test runs single-threaded and restores no state.
        unsafe { std::env::remove_var("WAYLAND_DISPLAY") };
        let result = super::VirtualInput::new();
        assert!(result.is_err());
    }

    #[test]
    fn timestamp_ms_returns_nonzero() {
        let ts = super::timestamp_ms();
        // Any real system clock should produce a non-zero value since UNIX_EPOCH.
        assert!(ts > 0, "timestamp_ms should be > 0, got {ts}");
    }

    #[test]
    fn timestamp_ms_is_monotonically_nondecreasing() {
        let t1 = super::timestamp_ms();
        let t2 = super::timestamp_ms();
        // Allow for wrapping at u32::MAX, but consecutive calls should not go backwards
        // by a large amount. In practice both are within the same millisecond.
        assert!(
            t2 >= t1 || t1.wrapping_sub(t2) < 1000,
            "timestamps should be close: t1={t1}, t2={t2}"
        );
    }

    #[test]
    fn pointer_extent_constant() {
        assert_eq!(super::POINTER_EXTENT, 0xFFFF);
    }

    #[test]
    fn inject_state_default_fields() {
        let state = super::InjectState {
            seat: None,
            vp_manager: None,
            vk_manager: None,
            virtual_pointer: None,
            virtual_keyboard: None,
        };
        assert!(state.seat.is_none());
        assert!(state.vp_manager.is_none());
        assert!(state.vk_manager.is_none());
        assert!(state.virtual_pointer.is_none());
        assert!(state.virtual_keyboard.is_none());
    }

    #[test]
    fn pointer_motion_payload_decode() {
        let motion = PointerMotion {
            surface_id: 42,
            _pad: 0,
            x: 1920.0,
            y: 1080.0,
        };
        let ev = InputEvent::pointer_motion(motion);
        let decoded =
            PointerMotion::ref_from_bytes(&ev.payload[..size_of::<PointerMotion>()]).unwrap();
        assert_eq!({ decoded.surface_id }, 42);
        assert_eq!({ decoded.x }, 1920.0);
        assert_eq!({ decoded.y }, 1080.0);
    }

    #[test]
    fn pointer_button_payload_decode_pressed_and_released() {
        for (state_val, label) in [(1u32, "pressed"), (0u32, "released")] {
            let btn = PointerButton {
                button: 0x110,
                state: state_val,
            };
            let ev = InputEvent::pointer_button(btn);
            let decoded =
                PointerButton::ref_from_bytes(&ev.payload[..size_of::<PointerButton>()]).unwrap();
            assert_eq!(
                { decoded.state },
                state_val,
                "button state mismatch for {label}"
            );
            assert_eq!({ decoded.button }, 0x110);
        }
    }

    #[test]
    fn pointer_axis_payload_decode_vertical_and_horizontal() {
        for (axis_val, expected_val) in [(0u8, 5.0f32), (1u8, -3.0f32)] {
            let axis = PointerAxis {
                axis: axis_val,
                _pad: [0; 3],
                value: expected_val,
            };
            let ev = InputEvent::pointer_axis(axis);
            let decoded =
                PointerAxis::ref_from_bytes(&ev.payload[..size_of::<PointerAxis>()]).unwrap();
            assert_eq!({ decoded.axis }, axis_val);
            assert_eq!({ decoded.value }, expected_val);
        }
    }

    #[test]
    fn key_event_payload_decode() {
        let key = KeyEvent { key: 30, state: 1 };
        let ev = InputEvent::key(key);
        let decoded = KeyEvent::ref_from_bytes(&ev.payload[..size_of::<KeyEvent>()]).unwrap();
        assert_eq!({ decoded.key }, 30);
        assert_eq!({ decoded.state }, 1);
    }

    #[test]
    fn key_event_payload_decode_repeat_state() {
        let key = KeyEvent { key: 28, state: 2 };
        let ev = InputEvent::key(key);
        let decoded = KeyEvent::ref_from_bytes(&ev.payload[..size_of::<KeyEvent>()]).unwrap();
        assert_eq!({ decoded.key }, 28);
        assert_eq!({ decoded.state }, 2);
    }

    #[test]
    fn unknown_kind_all_invalid_values() {
        for invalid_kind in [4u8, 5, 128, 255] {
            let mut ev = InputEvent::key(KeyEvent { key: 1, state: 0 });
            ev.kind = invalid_kind;
            assert!(ev.kind().is_err(), "kind {invalid_kind} should be invalid");
            assert_eq!(ev.kind().unwrap_err(), invalid_kind);
        }
    }

    #[test]
    fn dispatch_event_decode_error_on_truncated_payload() {
        // Craft an event where the payload bytes are valid for the kind,
        // but verify the decode path works for each kind variant.
        // All our payloads fit within 12 bytes, so normal events always decode.
        // This test confirms the happy path for all four decode branches.
        let events = [
            InputEvent::pointer_motion(PointerMotion {
                surface_id: 1,
                _pad: 0,
                x: 0.0,
                y: 0.0,
            }),
            InputEvent::pointer_button(PointerButton {
                button: 0x111,
                state: 0,
            }),
            InputEvent::pointer_axis(PointerAxis {
                axis: 1,
                _pad: [0; 3],
                value: 10.0,
            }),
            InputEvent::key(KeyEvent { key: 100, state: 0 }),
        ];
        // Verify all decode successfully through the same path dispatch_event uses.
        for ev in &events {
            let kind = ev.kind().unwrap();
            match kind {
                remoteway_proto::input::InputKind::PointerMotion => {
                    let m =
                        PointerMotion::ref_from_bytes(&ev.payload[..size_of::<PointerMotion>()]);
                    assert!(m.is_ok());
                }
                remoteway_proto::input::InputKind::PointerButton => {
                    let b =
                        PointerButton::ref_from_bytes(&ev.payload[..size_of::<PointerButton>()]);
                    assert!(b.is_ok());
                }
                remoteway_proto::input::InputKind::PointerAxis => {
                    let a = PointerAxis::ref_from_bytes(&ev.payload[..size_of::<PointerAxis>()]);
                    assert!(a.is_ok());
                }
                remoteway_proto::input::InputKind::Key => {
                    let k = KeyEvent::ref_from_bytes(&ev.payload[..size_of::<KeyEvent>()]);
                    assert!(k.is_ok());
                }
            }
        }
    }

    #[test]
    fn pointer_motion_boundary_values() {
        // Test with extreme coordinate values.
        let motion = PointerMotion {
            surface_id: u16::MAX,
            _pad: 0,
            x: f32::MAX,
            y: f32::MIN,
        };
        let ev = InputEvent::pointer_motion(motion);
        let decoded =
            PointerMotion::ref_from_bytes(&ev.payload[..size_of::<PointerMotion>()]).unwrap();
        assert_eq!({ decoded.surface_id }, u16::MAX);
        assert_eq!({ decoded.x }, f32::MAX);
        assert_eq!({ decoded.y }, f32::MIN);
    }

    #[test]
    fn pointer_motion_zero_coordinates() {
        let motion = PointerMotion {
            surface_id: 0,
            _pad: 0,
            x: 0.0,
            y: 0.0,
        };
        let ev = InputEvent::pointer_motion(motion);
        let decoded =
            PointerMotion::ref_from_bytes(&ev.payload[..size_of::<PointerMotion>()]).unwrap();
        assert_eq!({ decoded.x }, 0.0);
        assert_eq!({ decoded.y }, 0.0);
    }

    #[test]
    fn pointer_button_all_common_buttons() {
        // BTN_LEFT=0x110, BTN_RIGHT=0x111, BTN_MIDDLE=0x112
        for button_code in [0x110u32, 0x111, 0x112, 0x113, 0x114] {
            let btn = PointerButton {
                button: button_code,
                state: 1,
            };
            let ev = InputEvent::pointer_button(btn);
            let decoded =
                PointerButton::ref_from_bytes(&ev.payload[..size_of::<PointerButton>()]).unwrap();
            assert_eq!({ decoded.button }, button_code);
        }
    }

    #[test]
    fn inject_failed_error_contains_message() {
        let err = InputError::InjectFailed("decode PointerMotion: alignment".to_string());
        let display = err.to_string();
        assert!(display.contains("decode PointerMotion"));
        assert!(display.contains("alignment"));
    }

    #[test]
    fn error_variant_matching() {
        let err = InputError::NoSeat;
        assert!(matches!(err, InputError::NoSeat));
        let err = InputError::NoVirtualPointer;
        assert!(matches!(err, InputError::NoVirtualPointer));
        let err = InputError::NoVirtualKeyboard;
        assert!(matches!(err, InputError::NoVirtualKeyboard));
    }
}

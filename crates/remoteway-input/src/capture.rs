use wayland_client::protocol::{wl_keyboard, wl_pointer, wl_registry, wl_seat};
use wayland_client::{Connection, Dispatch, QueueHandle, WEnum};

use remoteway_proto::input::{InputEvent, KeyEvent, PointerAxis, PointerButton, PointerMotion};

use crate::error::InputError;

/// Captures pointer and keyboard events from the local Wayland compositor.
///
/// Used on the client side to intercept user input and serialize it
/// into `InputEvent` for transmission to the remote server.
pub struct InputCapture {
    _conn: Connection,
    state: CaptureState,
    event_queue: wayland_client::EventQueue<CaptureState>,
}

struct CaptureState {
    seat: Option<wl_seat::WlSeat>,
    pointer: Option<wl_pointer::WlPointer>,
    keyboard: Option<wl_keyboard::WlKeyboard>,
    pending_events: Vec<InputEvent>,
    /// Surface ID for pointer events (simplified to 0 for remote desktop use).
    surface_id: u16,
    capabilities_received: bool,
}

impl InputCapture {
    /// Connect to the local Wayland compositor and bind pointer/keyboard.
    pub fn new() -> Result<Self, InputError> {
        let conn = Connection::connect_to_env()?;
        let display = conn.display();
        let mut event_queue = conn.new_event_queue::<CaptureState>();
        let qh = event_queue.handle();

        let mut state = CaptureState {
            seat: None,
            pointer: None,
            keyboard: None,
            pending_events: Vec::with_capacity(64),
            surface_id: 0,
            capabilities_received: false,
        };

        let _ = display.get_registry(&qh, ()); // INTENTIONAL: WlRegistry managed by event queue

        // First roundtrip: discover globals (wl_seat).
        let _ = event_queue.roundtrip(&mut state)?; // INTENTIONAL: dispatch count irrelevant

        if state.seat.is_none() {
            return Err(InputError::NoSeat);
        }

        // Second roundtrip: receive seat capabilities and create pointer/keyboard.
        let _ = event_queue.roundtrip(&mut state)?; // INTENTIONAL: dispatch count irrelevant

        Ok(Self {
            _conn: conn,
            state,
            event_queue,
        })
    }

    /// Poll for pending input events (non-blocking dispatch).
    ///
    /// Returns a slice of `InputEvent`s accumulated during this dispatch round.
    /// The internal buffer is reused between calls.
    pub fn poll_events(&mut self) -> Result<&[InputEvent], InputError> {
        self.state.pending_events.clear();

        // Dispatch pending events without blocking.
        let _ = self.event_queue.dispatch_pending(&mut self.state)?; // INTENTIONAL: dispatch count irrelevant

        // Also try to read from the socket if data is available.
        if let Some(guard) = self.event_queue.prepare_read() {
            // Non-blocking read: returns Ok(n) with events read, or Err on I/O error.
            // WouldBlock is expected when no data is available.
            match guard.read() {
                Ok(_) => {
                    let _ = self.event_queue.dispatch_pending(&mut self.state)?; // INTENTIONAL: dispatch count irrelevant
                }
                Err(wayland_client::backend::WaylandError::Io(ref e))
                    if e.kind() == std::io::ErrorKind::WouldBlock => {}
                Err(e) => {
                    return Err(InputError::InjectFailed(format!("wayland read: {e}")));
                }
            }
        }

        Ok(&self.state.pending_events)
    }

    /// Blocking poll — waits for at least one event.
    pub fn poll_events_blocking(&mut self) -> Result<&[InputEvent], InputError> {
        self.state.pending_events.clear();
        let _ = self.event_queue.blocking_dispatch(&mut self.state)?; // INTENTIONAL: dispatch count irrelevant
        Ok(&self.state.pending_events)
    }
}

// --- Wayland dispatch implementations ---

impl Dispatch<wl_registry::WlRegistry, ()> for CaptureState {
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
            && interface.as_str() == "wl_seat"
        {
            state.seat = Some(registry.bind(name, version.min(8), qh, ()));
        }
    }
}

impl Dispatch<wl_seat::WlSeat, ()> for CaptureState {
    fn event(
        state: &mut Self,
        seat: &wl_seat::WlSeat,
        event: wl_seat::Event,
        _: &(),
        _conn: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if let wl_seat::Event::Capabilities {
            capabilities: WEnum::Value(caps),
        } = event
        {
            if caps.contains(wl_seat::Capability::Pointer) && state.pointer.is_none() {
                state.pointer = Some(seat.get_pointer(qh, ()));
            }
            if caps.contains(wl_seat::Capability::Keyboard) && state.keyboard.is_none() {
                state.keyboard = Some(seat.get_keyboard(qh, ()));
            }
            state.capabilities_received = true;
        }
    }
}

impl Dispatch<wl_pointer::WlPointer, ()> for CaptureState {
    fn event(
        state: &mut Self,
        _proxy: &wl_pointer::WlPointer,
        event: wl_pointer::Event,
        _: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        match event {
            wl_pointer::Event::Motion {
                surface_x,
                surface_y,
                ..
            } => {
                state
                    .pending_events
                    .push(InputEvent::pointer_motion(PointerMotion {
                        surface_id: state.surface_id,
                        _pad: 0,
                        x: surface_x as f32,
                        y: surface_y as f32,
                    }));
            }
            wl_pointer::Event::Button {
                button,
                state: WEnum::Value(s),
                ..
            } => {
                let state_val = match s {
                    wl_pointer::ButtonState::Released => 0u32,
                    wl_pointer::ButtonState::Pressed => 1u32,
                    _ => return,
                };
                state
                    .pending_events
                    .push(InputEvent::pointer_button(PointerButton {
                        button,
                        state: state_val,
                    }));
            }
            wl_pointer::Event::Axis {
                axis: WEnum::Value(a),
                value,
                ..
            } => {
                let axis_val = match a {
                    wl_pointer::Axis::VerticalScroll => 0u8,
                    wl_pointer::Axis::HorizontalScroll => 1u8,
                    _ => return,
                };
                state
                    .pending_events
                    .push(InputEvent::pointer_axis(PointerAxis {
                        axis: axis_val,
                        _pad: [0; 3],
                        value: value as f32,
                    }));
            }
            // Enter, Leave, Frame, AxisSource, etc. — not forwarded.
            _ => {}
        }
    }
}

impl Dispatch<wl_keyboard::WlKeyboard, ()> for CaptureState {
    fn event(
        state: &mut Self,
        _proxy: &wl_keyboard::WlKeyboard,
        event: wl_keyboard::Event,
        _: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        if let wl_keyboard::Event::Key {
            key,
            state: WEnum::Value(s),
            ..
        } = event
        {
            let state_val = match s {
                wl_keyboard::KeyState::Released => 0u32,
                wl_keyboard::KeyState::Pressed => 1u32,
                _ => return,
            };
            state.pending_events.push(InputEvent::key(KeyEvent {
                key,
                state: state_val,
            }));
        }
    }
}

#[cfg(test)]
mod tests {
    use remoteway_proto::input::{
        InputEvent, InputKind, KeyEvent, PointerAxis, PointerButton, PointerMotion,
    };
    use zerocopy::FromBytes;

    #[test]
    fn pointer_motion_event_creation() {
        let ev = InputEvent::pointer_motion(PointerMotion {
            surface_id: 0,
            _pad: 0,
            x: 100.5,
            y: 200.3,
        });
        assert_eq!(ev.kind().unwrap(), InputKind::PointerMotion);
    }

    #[test]
    fn pointer_button_event_creation() {
        let ev = InputEvent::pointer_button(PointerButton {
            button: 0x110,
            state: 1,
        });
        assert_eq!(ev.kind().unwrap(), InputKind::PointerButton);
    }

    #[test]
    fn pointer_axis_event_creation() {
        let ev = InputEvent::pointer_axis(PointerAxis {
            axis: 0,
            _pad: [0; 3],
            value: -15.0,
        });
        assert_eq!(ev.kind().unwrap(), InputKind::PointerAxis);
    }

    #[test]
    fn key_event_creation() {
        let ev = InputEvent::key(KeyEvent { key: 30, state: 1 });
        assert_eq!(ev.kind().unwrap(), InputKind::Key);
    }

    #[test]
    fn capture_new_fails_without_compositor() {
        // SAFETY: this test runs single-threaded and restores no state.
        unsafe { std::env::remove_var("WAYLAND_DISPLAY") };
        let result = super::InputCapture::new();
        assert!(result.is_err());
    }

    #[test]
    fn capture_state_default_fields() {
        let state = super::CaptureState {
            seat: None,
            pointer: None,
            keyboard: None,
            pending_events: Vec::new(),
            surface_id: 0,
            capabilities_received: false,
        };
        assert!(state.seat.is_none());
        assert!(state.pointer.is_none());
        assert!(state.keyboard.is_none());
        assert!(state.pending_events.is_empty());
        assert_eq!(state.surface_id, 0);
        assert!(!state.capabilities_received);
    }

    #[test]
    fn capture_state_pending_events_pre_allocated() {
        let state = super::CaptureState {
            seat: None,
            pointer: None,
            keyboard: None,
            pending_events: Vec::with_capacity(64),
            surface_id: 0,
            capabilities_received: false,
        };
        assert!(state.pending_events.capacity() >= 64);
    }

    #[test]
    fn pointer_motion_event_preserves_surface_id() {
        for surface_id in [0u16, 1, 42, u16::MAX] {
            let ev = InputEvent::pointer_motion(PointerMotion {
                surface_id,
                _pad: 0,
                x: 100.0,
                y: 200.0,
            });
            let decoded =
                PointerMotion::ref_from_bytes(&ev.payload[..size_of::<PointerMotion>()]).unwrap();
            assert_eq!({ decoded.surface_id }, surface_id);
        }
    }

    #[test]
    fn pointer_button_state_mapping() {
        // Released = 0, Pressed = 1 — matches the wl_pointer::ButtonState mapping in capture.
        let released = InputEvent::pointer_button(PointerButton {
            button: 0x110,
            state: 0,
        });
        let pressed = InputEvent::pointer_button(PointerButton {
            button: 0x110,
            state: 1,
        });
        let decoded_released =
            PointerButton::ref_from_bytes(&released.payload[..size_of::<PointerButton>()]).unwrap();
        let decoded_pressed =
            PointerButton::ref_from_bytes(&pressed.payload[..size_of::<PointerButton>()]).unwrap();
        assert_eq!({ decoded_released.state }, 0);
        assert_eq!({ decoded_pressed.state }, 1);
    }

    #[test]
    fn pointer_axis_vertical_is_zero_horizontal_is_one() {
        // axis: 0 = vertical, 1 = horizontal — matching the capture Dispatch logic.
        let vertical = InputEvent::pointer_axis(PointerAxis {
            axis: 0,
            _pad: [0; 3],
            value: 5.0,
        });
        let horizontal = InputEvent::pointer_axis(PointerAxis {
            axis: 1,
            _pad: [0; 3],
            value: -3.0,
        });
        let decoded_v =
            PointerAxis::ref_from_bytes(&vertical.payload[..size_of::<PointerAxis>()]).unwrap();
        let decoded_h =
            PointerAxis::ref_from_bytes(&horizontal.payload[..size_of::<PointerAxis>()]).unwrap();
        assert_eq!({ decoded_v.axis }, 0);
        assert_eq!({ decoded_h.axis }, 1);
        assert_eq!({ decoded_v.value }, 5.0);
        assert_eq!({ decoded_h.value }, -3.0);
    }

    #[test]
    fn key_event_state_mapping() {
        // Released = 0, Pressed = 1 — matches the capture keyboard Dispatch logic.
        let released = InputEvent::key(KeyEvent { key: 30, state: 0 });
        let pressed = InputEvent::key(KeyEvent { key: 30, state: 1 });
        let decoded_released =
            KeyEvent::ref_from_bytes(&released.payload[..size_of::<KeyEvent>()]).unwrap();
        let decoded_pressed =
            KeyEvent::ref_from_bytes(&pressed.payload[..size_of::<KeyEvent>()]).unwrap();
        assert_eq!({ decoded_released.state }, 0);
        assert_eq!({ decoded_pressed.state }, 1);
    }

    #[test]
    fn pending_events_push_and_clear() {
        let mut state = super::CaptureState {
            seat: None,
            pointer: None,
            keyboard: None,
            pending_events: Vec::new(),
            surface_id: 0,
            capabilities_received: false,
        };
        // Simulate what the Dispatch handlers do.
        state
            .pending_events
            .push(InputEvent::pointer_motion(PointerMotion {
                surface_id: 0,
                _pad: 0,
                x: 10.0,
                y: 20.0,
            }));
        state
            .pending_events
            .push(InputEvent::key(KeyEvent { key: 30, state: 1 }));
        assert_eq!(state.pending_events.len(), 2);

        // Clear as poll_events does.
        state.pending_events.clear();
        assert!(state.pending_events.is_empty());
    }

    #[test]
    fn pointer_motion_subpixel_precision() {
        // Verify that sub-pixel coordinates survive the f64->f32 cast used in capture.
        let x: f64 = 123.456;
        let y: f64 = 789.012;
        let ev = InputEvent::pointer_motion(PointerMotion {
            surface_id: 0,
            _pad: 0,
            x: x as f32,
            y: y as f32,
        });
        let decoded =
            PointerMotion::ref_from_bytes(&ev.payload[..size_of::<PointerMotion>()]).unwrap();
        // f32 precision: within 0.001 of the original.
        assert!(({ decoded.x } - 123.456f32).abs() < 0.001);
        assert!(({ decoded.y } - 789.012f32).abs() < 0.001);
    }

    #[test]
    fn many_events_in_pending_buffer() {
        let mut state = super::CaptureState {
            seat: None,
            pointer: None,
            keyboard: None,
            pending_events: Vec::with_capacity(64),
            surface_id: 0,
            capabilities_received: false,
        };
        // Push a realistic burst of events.
        for i in 0..50 {
            state
                .pending_events
                .push(InputEvent::pointer_motion(PointerMotion {
                    surface_id: 0,
                    _pad: 0,
                    x: i as f32,
                    y: (i * 2) as f32,
                }));
        }
        assert_eq!(state.pending_events.len(), 50);
        // Verify first and last.
        let first = PointerMotion::ref_from_bytes(
            &state.pending_events[0].payload[..size_of::<PointerMotion>()],
        )
        .unwrap();
        let last = PointerMotion::ref_from_bytes(
            &state.pending_events[49].payload[..size_of::<PointerMotion>()],
        )
        .unwrap();
        assert_eq!({ first.x }, 0.0);
        assert_eq!({ last.x }, 49.0);
    }

    #[test]
    fn capture_state_capabilities_flag() {
        let mut state = super::CaptureState {
            seat: None,
            pointer: None,
            keyboard: None,
            pending_events: Vec::new(),
            surface_id: 0,
            capabilities_received: false,
        };
        assert!(!state.capabilities_received);
        state.capabilities_received = true;
        assert!(state.capabilities_received);
    }

    #[test]
    fn capture_state_surface_id_custom() {
        let state = super::CaptureState {
            seat: None,
            pointer: None,
            keyboard: None,
            pending_events: Vec::new(),
            surface_id: 7,
            capabilities_received: false,
        };
        assert_eq!(state.surface_id, 7);
    }

    #[test]
    fn axis_value_preserves_negative() {
        let ev = InputEvent::pointer_axis(PointerAxis {
            axis: 0,
            _pad: [0; 3],
            value: -100.5,
        });
        let decoded = PointerAxis::ref_from_bytes(&ev.payload[..size_of::<PointerAxis>()]).unwrap();
        assert_eq!({ decoded.value }, -100.5);
    }

    #[test]
    fn axis_cast_f64_to_f32() {
        // The capture code does `value as f32` where value is f64 from the Wayland event.
        let f64_value: f64 = 15.5;
        let as_f32 = f64_value as f32;
        let ev = InputEvent::pointer_axis(PointerAxis {
            axis: 0,
            _pad: [0; 3],
            value: as_f32,
        });
        let decoded = PointerAxis::ref_from_bytes(&ev.payload[..size_of::<PointerAxis>()]).unwrap();
        assert_eq!({ decoded.value }, 15.5f32);
    }
}

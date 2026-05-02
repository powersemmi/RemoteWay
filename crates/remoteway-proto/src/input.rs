use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

/// Pointer motion in surface-local coordinates (pixels, f32 for sub-pixel precision).
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct PointerMotion {
    /// Target surface ID.
    pub surface_id: u16,
    /// Alignment padding; always zero.
    pub _pad: u16,
    /// X coordinate in surface-local pixels.
    pub x: f32,
    /// Y coordinate in surface-local pixels.
    pub y: f32,
}

/// Pointer button press or release.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct PointerButton {
    /// Linux evdev button code (`BTN_LEFT` = 0x110, etc.).
    pub button: u32,
    /// 1 = pressed, 0 = released.
    pub state: u32,
}

/// Pointer scroll (axis) event.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct PointerAxis {
    /// 0 = vertical, 1 = horizontal.
    pub axis: u8,
    /// Alignment padding; always zero.
    pub _pad: [u8; 3],
    /// Scroll delta (positive = down/right).
    pub value: f32,
}

/// Key press or release.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct KeyEvent {
    /// Linux evdev key code.
    pub key: u32,
    /// 1 = pressed, 0 = released, 2 = repeat.
    pub state: u32,
}

/// Discriminant for the union inside [`InputEvent`].
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputKind {
    /// Pointer motion event.
    PointerMotion = 0,
    /// Pointer button press or release.
    PointerButton = 1,
    /// Pointer scroll (axis) event.
    PointerAxis = 2,
    /// Keyboard key press or release.
    Key = 3,
}

impl TryFrom<u8> for InputKind {
    type Error = u8;
    fn try_from(v: u8) -> Result<Self, u8> {
        match v {
            0 => Ok(Self::PointerMotion),
            1 => Ok(Self::PointerButton),
            2 => Ok(Self::PointerAxis),
            3 => Ok(Self::Key),
            other => Err(other),
        }
    }
}

/// Fixed-size input event (16 bytes) transmitted on the input stream.
///
/// Layout: kind(1) + _pad(3) + payload(12) = 16 bytes.
/// `payload` is large enough to hold the widest sub-type (`PointerMotion` = 12 bytes).
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct InputEvent {
    /// Discriminant from [`InputKind`].
    pub kind: u8,
    /// Alignment padding; always zero.
    pub _pad: [u8; 3],
    /// Sub-type payload (size varies by kind).
    pub payload: [u8; 12],
}

const _: () = assert!(size_of::<InputEvent>() == 16);

impl InputEvent {
    /// Pack a sub-type's bytes into the fixed-size payload.
    fn from_bytes_padded(kind: InputKind, src: &[u8]) -> Self {
        let mut payload = [0u8; 12];
        payload[..src.len()].copy_from_slice(src);
        Self {
            kind: kind as u8,
            _pad: [0; 3],
            payload,
        }
    }

    /// Create an `InputEvent` from a [`PointerMotion`].
    #[must_use]
    pub fn pointer_motion(motion: PointerMotion) -> Self {
        Self::from_bytes_padded(InputKind::PointerMotion, IntoBytes::as_bytes(&motion))
    }

    /// Create an `InputEvent` from a [`PointerButton`].
    #[must_use]
    pub fn pointer_button(btn: PointerButton) -> Self {
        Self::from_bytes_padded(InputKind::PointerButton, IntoBytes::as_bytes(&btn))
    }

    /// Create an `InputEvent` from a [`PointerAxis`].
    #[must_use]
    pub fn pointer_axis(axis: PointerAxis) -> Self {
        Self::from_bytes_padded(InputKind::PointerAxis, IntoBytes::as_bytes(&axis))
    }

    /// Create an `InputEvent` from a [`KeyEvent`].
    #[must_use]
    pub fn key(key: KeyEvent) -> Self {
        Self::from_bytes_padded(InputKind::Key, IntoBytes::as_bytes(&key))
    }

    /// Decode the `kind` byte into an [`InputKind`] discriminant.
    pub fn kind(&self) -> Result<InputKind, u8> {
        InputKind::try_from(self.kind)
    }
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use zerocopy::IntoBytes;

    use super::*;

    #[test]
    fn input_event_is_16_bytes() {
        assert_eq!(std::mem::size_of::<InputEvent>(), 16);
    }

    #[test]
    fn pointer_motion_round_trip() {
        let motion = PointerMotion {
            surface_id: 2,
            _pad: 0,
            x: 123.5,
            y: 456.0,
        };
        let ev = InputEvent::pointer_motion(motion);
        assert_eq!(ev.kind().unwrap(), InputKind::PointerMotion);
        let decoded =
            PointerMotion::ref_from_bytes(&ev.payload[..size_of::<PointerMotion>()]).unwrap();
        assert_eq!({ decoded.surface_id }, 2);
        assert_eq!({ decoded.x }, 123.5);
    }

    #[test]
    fn key_event_round_trip() {
        let key = KeyEvent { key: 30, state: 1 };
        let ev = InputEvent::key(key);
        assert_eq!(ev.kind().unwrap(), InputKind::Key);
        let decoded = KeyEvent::ref_from_bytes(&ev.payload[..size_of::<KeyEvent>()]).unwrap();
        assert_eq!({ decoded.key }, 30);
        assert_eq!({ decoded.state }, 1);
    }

    #[test]
    fn unknown_kind_returns_err() {
        let mut ev = InputEvent::key(KeyEvent { key: 1, state: 0 });
        ev.kind = 99;
        assert!(ev.kind().is_err());
    }

    #[test]
    fn all_events_serialize_to_bytes() {
        let events = [
            InputEvent::pointer_motion(PointerMotion {
                surface_id: 0,
                _pad: 0,
                x: 0.0,
                y: 0.0,
            }),
            InputEvent::pointer_button(PointerButton {
                button: 0x110,
                state: 1,
            }),
            InputEvent::pointer_axis(PointerAxis {
                axis: 0,
                _pad: [0; 3],
                value: 1.0,
            }),
            InputEvent::key(KeyEvent { key: 1, state: 1 }),
        ];
        for ev in &events {
            assert_eq!(ev.as_bytes().len(), 16);
        }
    }

    #[test]
    fn pointer_button_btn_left_round_trip() {
        let btn = PointerButton {
            button: 0x110, // BTN_LEFT
            state: 1,
        };
        let ev = InputEvent::pointer_button(btn);
        assert_eq!(ev.kind().unwrap(), InputKind::PointerButton);
        let decoded =
            PointerButton::ref_from_bytes(&ev.payload[..size_of::<PointerButton>()]).unwrap();
        assert_eq!({ decoded.button }, 0x110);
        assert_eq!({ decoded.state }, 1);
    }

    #[test]
    fn pointer_button_btn_right_released() {
        let btn = PointerButton {
            button: 0x111, // BTN_RIGHT
            state: 0,
        };
        let ev = InputEvent::pointer_button(btn);
        let decoded =
            PointerButton::ref_from_bytes(&ev.payload[..size_of::<PointerButton>()]).unwrap();
        assert_eq!({ decoded.button }, 0x111);
        assert_eq!({ decoded.state }, 0);
    }

    #[test]
    fn pointer_axis_vertical_round_trip() {
        let axis = PointerAxis {
            axis: 0,
            _pad: [0; 3],
            value: 15.5,
        };
        let ev = InputEvent::pointer_axis(axis);
        assert_eq!(ev.kind().unwrap(), InputKind::PointerAxis);
        let decoded = PointerAxis::ref_from_bytes(&ev.payload[..size_of::<PointerAxis>()]).unwrap();
        assert_eq!({ decoded.axis }, 0);
        assert_eq!({ decoded.value }, 15.5);
    }

    #[test]
    fn pointer_axis_horizontal_negative() {
        let axis = PointerAxis {
            axis: 1,
            _pad: [0; 3],
            value: -3.0,
        };
        let ev = InputEvent::pointer_axis(axis);
        let decoded = PointerAxis::ref_from_bytes(&ev.payload[..size_of::<PointerAxis>()]).unwrap();
        assert_eq!({ decoded.axis }, 1);
        assert_eq!({ decoded.value }, -3.0);
    }

    #[test]
    fn unified_as_bytes_from_bytes_all_kinds() {
        let events = [
            InputEvent::pointer_motion(PointerMotion {
                surface_id: 1,
                _pad: 0,
                x: 10.0,
                y: 20.0,
            }),
            InputEvent::pointer_button(PointerButton {
                button: 0x110,
                state: 1,
            }),
            InputEvent::pointer_axis(PointerAxis {
                axis: 0,
                _pad: [0; 3],
                value: 5.0,
            }),
            InputEvent::key(KeyEvent { key: 28, state: 2 }),
        ];
        let expected_kinds = [
            InputKind::PointerMotion,
            InputKind::PointerButton,
            InputKind::PointerAxis,
            InputKind::Key,
        ];
        for (ev, expected) in events.iter().zip(expected_kinds.iter()) {
            let bytes = ev.as_bytes();
            let decoded = InputEvent::ref_from_bytes(bytes).unwrap();
            assert_eq!(decoded.kind().unwrap(), *expected);
            assert_eq!(decoded.as_bytes(), ev.as_bytes());
        }
    }

    #[test]
    fn sub_struct_sizes() {
        assert_eq!(size_of::<PointerMotion>(), 12);
        assert_eq!(size_of::<PointerButton>(), 8);
        assert_eq!(size_of::<PointerAxis>(), 8);
        assert_eq!(size_of::<KeyEvent>(), 8);
    }
}

//! libei (Emulated Input) backend for GNOME input injection.
//!
//! Requires the `gnome` feature flag. Uses the `reis` crate to implement
//! the EI (Emulated Input) protocol for injecting pointer and keyboard
//! events on GNOME/Mutter compositors that don't support wlr-virtual-pointer.

use std::os::unix::net::UnixStream;

use reis::ei;
use reis::event::{DeviceCapability, EiConvertEventIterator, EiEvent};
use remoteway_proto::input::{
    InputEvent, InputKind, KeyEvent, PointerAxis, PointerButton, PointerMotion,
};
use zerocopy::FromBytes;

use crate::error::InputError;

/// libei-based virtual input backend for GNOME.
///
/// Connects to the EI server (typically Mutter) via an EI socket and
/// creates emulated pointer/keyboard devices for input injection.
pub struct EiInput {
    connection: reis::event::Connection,
    events: EiConvertEventIterator,
    device: Option<reis::event::Device>,
}

impl EiInput {
    /// Connect to the EI server and perform handshake.
    ///
    /// The `socket` is typically obtained from xdg-desktop-portal's
    /// RemoteDesktop interface via `ConnectToEIS`.
    pub fn new(socket: UnixStream) -> Result<Self, InputError> {
        let context = ei::Context::new(socket)
            .map_err(|e| InputError::Protocol(format!("EI context creation failed: {e}")))?;

        // Perform blocking handshake as a sender (input injector).
        let (connection, events) = context
            .handshake_blocking("remoteway", ei::handshake::ContextType::Sender)
            .map_err(|e| InputError::Protocol(format!("EI handshake failed: {e}")))?;

        Ok(Self {
            connection,
            events,
            device: None,
        })
    }

    /// Process pending EI events to discover devices.
    pub fn dispatch(&mut self) -> Result<(), InputError> {
        for event in self.events.by_ref() {
            let event = event.map_err(|e| InputError::Protocol(format!("EI event error: {e}")))?;
            match event {
                EiEvent::DeviceAdded(added) => {
                    let dev = added.device;
                    if dev.has_capability(DeviceCapability::Pointer)
                        || dev.has_capability(DeviceCapability::PointerAbsolute)
                        || dev.has_capability(DeviceCapability::Keyboard)
                    {
                        // Start emulating on this device.
                        let serial = self.connection.serial();
                        dev.device().start_emulating(serial, 0);
                        self.connection.flush().ok();
                        self.device = Some(dev);
                    }
                }
                EiEvent::DeviceRemoved(_) => {
                    self.device = None;
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Inject a single input event via the EI protocol.
    pub fn inject(&mut self, event: &InputEvent) -> Result<(), InputError> {
        let Some(ref device) = self.device else {
            return Err(InputError::Protocol("no EI device available".into()));
        };

        let kind = event.kind().map_err(InputError::UnknownInputKind)?;

        let serial = self.connection.serial();
        let timestamp = 0u64; // Use 0 for "now".

        match kind {
            InputKind::PointerMotion => {
                let motion =
                    PointerMotion::ref_from_bytes(&event.payload[..size_of::<PointerMotion>()])
                        .map_err(|_| {
                            InputError::Protocol("invalid PointerMotion payload".into())
                        })?;

                // Try absolute first, fall back to relative.
                if let Some(abs) = device.interface::<ei::PointerAbsolute>() {
                    abs.motion_absolute(motion.x, motion.y);
                } else if let Some(ptr) = device.interface::<ei::Pointer>() {
                    ptr.motion_relative(motion.x, motion.y);
                }
                device.device().frame(serial, timestamp);
            }
            InputKind::PointerButton => {
                let btn =
                    PointerButton::ref_from_bytes(&event.payload[..size_of::<PointerButton>()])
                        .map_err(|_| {
                            InputError::Protocol("invalid PointerButton payload".into())
                        })?;

                if let Some(button) = device.interface::<ei::Button>() {
                    let state = if btn.state != 0 {
                        ei::button::ButtonState::Press
                    } else {
                        ei::button::ButtonState::Released
                    };
                    button.button(btn.button, state);
                    device.device().frame(serial, timestamp);
                }
            }
            InputKind::PointerAxis => {
                let axis = PointerAxis::ref_from_bytes(&event.payload[..size_of::<PointerAxis>()])
                    .map_err(|_| InputError::Protocol("invalid PointerAxis payload".into()))?;

                if let Some(scroll) = device.interface::<ei::Scroll>() {
                    let (dx, dy) = if axis.axis == 0 {
                        (0.0f32, axis.value)
                    } else {
                        (axis.value, 0.0f32)
                    };
                    scroll.scroll(dx, dy);
                    device.device().frame(serial, timestamp);
                }
            }
            InputKind::Key => {
                let key = KeyEvent::ref_from_bytes(&event.payload[..size_of::<KeyEvent>()])
                    .map_err(|_| InputError::Protocol("invalid KeyEvent payload".into()))?;

                if let Some(kbd) = device.interface::<ei::Keyboard>() {
                    let state = if key.state != 0 {
                        ei::keyboard::KeyState::Press
                    } else {
                        ei::keyboard::KeyState::Released
                    };
                    kbd.key(key.key, state);
                    device.device().frame(serial, timestamp);
                }
            }
        }

        self.connection
            .flush()
            .map_err(|e| InputError::Protocol(format!("EI flush failed: {e}")))?;
        Ok(())
    }

    /// Inject a batch of input events.
    pub fn inject_batch(&mut self, events: &[InputEvent]) -> Result<(), InputError> {
        for event in events {
            self.inject(event)?;
        }
        Ok(())
    }
}

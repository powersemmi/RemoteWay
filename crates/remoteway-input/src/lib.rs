//! Input capture and injection for `RemoteWay`. Server-side virtual pointer/keyboard
//! injection; client-side input capture.

pub mod capture;
pub mod capture_thread;
pub mod error;
pub mod inject;
pub mod inject_thread;
pub mod keymap;
#[cfg(feature = "gnome")]
pub mod libei;

pub mod backend;
pub mod desktop_detect;
pub mod detect;
pub mod error;
pub mod ext_capture;
pub mod output;
#[cfg(feature = "gnome")]
pub mod pipewire_capture;
#[cfg(feature = "gnome")]
pub mod portal;
mod protocols;
pub mod screencopy;
pub mod shm;
pub mod thread;
pub mod toplevel;

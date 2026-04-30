pub mod cursor;
pub mod error;
pub mod presentation;
pub mod shm;
pub mod surface;
pub mod thread;

pub use error::DisplayError;
pub use shm::{DamageRegion, ShmFrameUploader};
pub use surface::{DisplayState, ManagedSurface, WaylandDisplay};
pub use thread::{DisplayFrame, DisplayThread, DisplayThreadConfig};

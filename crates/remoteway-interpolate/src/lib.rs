pub mod backend;
pub mod backends;
pub mod error;
pub mod interpolator;
pub mod manager;

pub use backend::{BackendDetector, BackendKind};
pub use error::InterpolateError;
pub use interpolator::{FrameInterpolator, GpuFrame, LinearBlendInterpolator};
pub use manager::InterpolationManager;

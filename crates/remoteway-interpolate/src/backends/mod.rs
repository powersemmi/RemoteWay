//! Pluggable interpolation backend implementations.
//!
//! Each backend is behind a feature gate and provides a concrete
//! [`FrameInterpolator`](crate::interpolator::FrameInterpolator).

#[cfg(feature = "wgpu-backend")]
pub mod wgpu_optical_flow;

#[cfg(any(feature = "fsr2", feature = "fsr3", feature = "nvidia-of"))]
pub mod vulkan_context;

#[cfg(feature = "fsr2")]
pub mod fsr2;

#[cfg(feature = "fsr3")]
pub mod fsr3;

#[cfg(feature = "nvidia-of")]
pub mod nvidia_optical_flow;

pub mod rife;

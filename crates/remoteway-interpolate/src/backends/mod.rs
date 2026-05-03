//! Pluggable interpolation backend implementations.

#[cfg(any(
    feature = "fsr2",
    feature = "fsr3",
    feature = "fsr2-rife",
    feature = "fsr2-native",
))]
pub mod vulkan_context;

#[cfg(feature = "fsr2")]
pub mod fsr2;

#[cfg(any(feature = "fsr2", feature = "fsr3", feature = "fsr2-rife"))]
pub mod fsr2_native;

#[cfg(feature = "fsr3")]
pub mod ffx_fg;

#[cfg(feature = "fsr3")]
pub mod fsr3;

#[cfg(feature = "fsr3")]
pub mod fsr3_frame_gen;

#[cfg(feature = "fsr2-rife")]
pub mod fsr2_rife;

#[cfg(any(feature = "rife", feature = "fsr2-rife"))]
pub mod rife;

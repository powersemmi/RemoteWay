//! GPU video encoding for `RemoteWay` via Vulkan Video.
//!
//! This crate defines the [`Encoder`] trait that all per-codec backends
//! implement. Backends share a single [`remoteway_vulkan::VulkanContext`] with
//! the rest of the GPU pipeline (capture → encode → frame generation) so frames
//! never leave device memory.
//!
//! The contract for backends is fixed by the trait and the integration test
//! suite in `tests/`. Backends MUST NOT alter the trait or the shared types;
//! they implement them and add codec-specific configuration alongside.

pub mod backends;
pub mod encoder;
pub mod error;

#[cfg(feature = "gpu-tests")]
pub mod test_support;

pub use encoder::{EncodeParams, EncodedFrame, Encoder, FrameKind, RateControl};
pub use error::EncodeError;

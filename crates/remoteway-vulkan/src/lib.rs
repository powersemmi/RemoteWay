//! Shared Vulkan foundation for `RemoteWay`'s GPU-backed crates.
//!
//! Provides a single `VulkanContext` that owns the Vulkan instance, physical
//! device selection, logical device, and queues used by both `remoteway-interpolate`
//! (FSR / frame generation) and `remoteway-encode` (Vulkan Video encode). Holding
//! these in one place lets the capture → encode → interpolate pipeline share a
//! single `VkDevice`, keeping frames on the GPU end-to-end without cross-device
//! copies.

pub mod context;
pub mod error;
pub mod video;

pub use context::{QueueRequest, VulkanContext};
pub use error::VulkanError;
pub use video::{VideoCodec, VideoEncodeCapabilities};

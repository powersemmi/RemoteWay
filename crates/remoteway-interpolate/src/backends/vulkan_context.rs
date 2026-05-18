//! Re-export of the shared `VulkanContext` from `remoteway-vulkan`.
//!
//! The implementation moved to `remoteway-vulkan` so the encode pipeline can
//! share a single `VkDevice` with FSR / frame generation. This module exists
//! only to keep existing intra-crate import paths (`super::vulkan_context::*`)
//! valid without touching every backend file.

pub use remoteway_vulkan::VulkanContext;

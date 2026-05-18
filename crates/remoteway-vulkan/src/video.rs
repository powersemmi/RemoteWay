//! Video encode capability discovery.
//!
//! Wraps the `VK_KHR_video_*` capability queries into a stable Rust enum/struct
//! surface that the encode crate consumes without re-touching `ash::vk` types.

use ash::vk;

/// Video codec supported by Vulkan Video encode extensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VideoCodec {
    H264,
    H265,
    Av1,
}

impl VideoCodec {
    /// Extension name string corresponding to this codec's encode extension.
    pub const fn encode_extension(self) -> &'static std::ffi::CStr {
        match self {
            Self::H264 => c"VK_KHR_video_encode_h264",
            Self::H265 => c"VK_KHR_video_encode_h265",
            Self::Av1 => c"VK_KHR_video_encode_av1",
        }
    }

    /// Maps to the Vulkan video codec operation flag advertised by the driver.
    #[must_use]
    pub fn codec_operation(self) -> vk::VideoCodecOperationFlagsKHR {
        match self {
            Self::H264 => vk::VideoCodecOperationFlagsKHR::ENCODE_H264,
            Self::H265 => vk::VideoCodecOperationFlagsKHR::ENCODE_H265,
            Self::Av1 => vk::VideoCodecOperationFlagsKHR::ENCODE_AV1,
        }
    }
}

/// Snapshot of what the selected physical device can encode.
///
/// Populated by `VulkanContext::probe_video_encode_capabilities`. Callers in
/// `remoteway-encode` use this to pick a codec and to validate user-supplied
/// `EncodeParams` (resolution bounds, supported rate-control modes, etc.).
#[derive(Debug, Clone)]
pub struct VideoEncodeCapabilities {
    pub codec: VideoCodec,
    pub max_coded_extent: (u32, u32),
    pub min_coded_extent: (u32, u32),
    pub picture_access_granularity: (u32, u32),
    pub max_dpb_slots: u32,
    pub max_active_reference_pictures: u32,
    pub rate_control_modes: vk::VideoEncodeRateControlModeFlagsKHR,
    pub supports_intra_refresh: bool,
    pub supports_quantization_map: bool,
}

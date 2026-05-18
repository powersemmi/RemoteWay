//! [`Encoder`] trait and the codec-agnostic types it operates on.
//!
//! Detailed semantics — backends and tests rely on these being precise:
//!
//! - All resolutions are in **pixels**, width × height.
//! - All frame inputs are in **`VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM`** (I420) or
//!   `VK_FORMAT_G8_B8R8_2PLANE_420_UNORM` (NV12). Backends declare which they
//!   accept via [`Encoder::accepted_input_format`]. Capture/colour conversion
//!   to YUV is the caller's responsibility (or a future
//!   `VK_VALVE_video_encode_rgb_conversion` path).
//! - [`Encoder::encode`] is **synchronous** w.r.t. the caller: it submits work
//!   and blocks until the bitstream is readable. A non-blocking variant will be
//!   added later when the transport crate needs back-pressure decoupling.
//! - Output bitstream is **Annex-B** (start-code prefixed NAL units for H.264/
//!   H.265; OBU-stream for AV1). The first encoded frame after `new` MUST be
//!   an IDR/keyframe and MUST be preceded by codec parameter sets
//!   (SPS/PPS/VPS for H.264/5; sequence header OBU for AV1).
//! - [`EncodedFrame::kind`] is authoritative: callers use it to drive transport
//!   priority and recovery on packet loss.

use ash::vk;
use remoteway_vulkan::{VideoCodec, VideoEncodeCapabilities};

use crate::EncodeError;

/// Parameters that fully describe an encoding session.
///
/// Backends validate against [`VideoEncodeCapabilities`] returned from the
/// Vulkan context. Invalid parameters surface as [`EncodeError::InvalidParams`].
///
/// [`VideoEncodeCapabilities`]: remoteway_vulkan::VideoEncodeCapabilities
/// [`EncodeError::InvalidParams`]: crate::EncodeError::InvalidParams
#[derive(Debug, Clone, Copy)]
pub struct EncodeParams {
    pub codec: VideoCodec,
    pub width: u32,
    pub height: u32,
    /// Frame rate numerator / denominator. Used by the encoder to fill HRD
    /// parameters and the rate-control feedback loop. For variable-rate capture
    /// (typical of screen content) set to an upper bound — actual delivery
    /// timing is the transport's responsibility.
    pub frame_rate: (u32, u32),
    pub rate_control: RateControl,
    /// Distance between intra-refresh waves (in frames). When `Some(n)`, the
    /// encoder uses `VK_KHR_video_encode_intra_refresh` to spread intra
    /// content across `n` frames instead of emitting periodic IDRs — critical
    /// for low-latency over lossy networks. When `None`, periodic IDRs are
    /// emitted at the codec's discretion.
    pub intra_refresh_period: Option<u32>,
}

impl EncodeParams {
    /// Validates these params against the device-reported capabilities.
    ///
    /// Each backend calls this at the top of `Encoder::new` so the validation
    /// rules stay consistent across H.264 / H.265 / AV1. Failures return
    /// [`EncodeError::InvalidParams`] with a human-readable reason.
    pub fn validate_against(&self, caps: &VideoEncodeCapabilities) -> Result<(), EncodeError> {
        if caps.codec != self.codec {
            return Err(EncodeError::InvalidParams(format!(
                "caps describe codec {:?}, params request {:?}",
                caps.codec, self.codec
            )));
        }
        if self.width == 0 || self.height == 0 {
            return Err(EncodeError::InvalidParams(format!(
                "resolution {}x{} has zero dimension",
                self.width, self.height
            )));
        }
        if self.width < caps.min_coded_extent.0
            || self.height < caps.min_coded_extent.1
            || self.width > caps.max_coded_extent.0
            || self.height > caps.max_coded_extent.1
        {
            return Err(EncodeError::InvalidParams(format!(
                "resolution {}x{} outside device range {:?}..={:?}",
                self.width, self.height, caps.min_coded_extent, caps.max_coded_extent
            )));
        }
        // Picture access granularity defines the multiple width/height must be aligned to
        // on the encoder side. We don't pad implicitly — caller must respect it.
        let (gx, gy) = caps.picture_access_granularity;
        if gx > 0 && self.width % gx != 0 {
            return Err(EncodeError::InvalidParams(format!(
                "width {} not aligned to picture_access_granularity {}",
                self.width, gx
            )));
        }
        if gy > 0 && self.height % gy != 0 {
            return Err(EncodeError::InvalidParams(format!(
                "height {} not aligned to picture_access_granularity {}",
                self.height, gy
            )));
        }
        let wanted_mode = self.rate_control.to_vk_mode();
        if !caps.rate_control_modes.contains(wanted_mode) {
            return Err(EncodeError::InvalidParams(format!(
                "rate control {:?} not in device-supported set {:?}",
                wanted_mode, caps.rate_control_modes
            )));
        }
        if self.intra_refresh_period.is_some() && !caps.supports_intra_refresh {
            return Err(EncodeError::InvalidParams(
                "intra_refresh_period set but device lacks VK_KHR_video_encode_intra_refresh".into(),
            ));
        }
        if let RateControl::ConstantQp { qp } = self.rate_control {
            // Codec-agnostic sanity: every codec's QP range fits in 0..=255.
            // Backends apply tighter per-codec bounds (e.g. H.265 is 0..=51).
            if qp > 255 {
                return Err(EncodeError::InvalidParams(format!("qp {qp} out of range")));
            }
        }
        if let (n, 0) = self.frame_rate {
            return Err(EncodeError::InvalidParams(format!(
                "frame_rate has zero denominator (numerator {n})"
            )));
        }
        Ok(())
    }
}

/// Rate-control strategy. Backends advertise which modes are supported via the
/// [`VideoEncodeCapabilities::rate_control_modes`] bitfield; passing an
/// unsupported mode fails [`Encoder::new`] with [`EncodeError::InvalidParams`].
///
/// [`VideoEncodeCapabilities::rate_control_modes`]: remoteway_vulkan::VideoEncodeCapabilities::rate_control_modes
/// [`EncodeError::InvalidParams`]: crate::EncodeError::InvalidParams
#[derive(Debug, Clone, Copy)]
pub enum RateControl {
    /// Constant quantization parameter. Lowest implementation complexity,
    /// no bandwidth target — output bitrate floats with content complexity.
    ConstantQp { qp: u32 },
    /// Constant bitrate (bits/sec). Encoder targets this average over a
    /// window defined by HRD parameters.
    Cbr { bitrate_bps: u64 },
    /// Variable bitrate with a peak ceiling. Average target plus peak cap.
    Vbr {
        average_bps: u64,
        peak_bps: u64,
    },
}

impl RateControl {
    /// Maps to the Vulkan rate-control mode flag advertised by the driver.
    ///
    /// Per Vulkan spec, `DISABLED` means "no rate control — the encoder uses
    /// a fixed QP supplied per-picture". So `ConstantQp` ↔ `DISABLED` is the
    /// intended mapping, not a workaround.
    #[must_use]
    pub fn to_vk_mode(self) -> vk::VideoEncodeRateControlModeFlagsKHR {
        match self {
            Self::ConstantQp { .. } => vk::VideoEncodeRateControlModeFlagsKHR::DISABLED,
            Self::Cbr { .. } => vk::VideoEncodeRateControlModeFlagsKHR::CBR,
            Self::Vbr { .. } => vk::VideoEncodeRateControlModeFlagsKHR::VBR,
        }
    }
}

/// Type of an encoded frame. Distinguishes recovery points for transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameKind {
    /// IDR — clean random-access point, decoder can resync without prior state.
    Idr,
    /// Non-IDR I-frame (rare in low-latency mode).
    I,
    /// P-frame — references prior frames in the DPB.
    P,
    /// Intra-refresh wave — partial intra content, decoder progressively
    /// recovers over `intra_refresh_period` frames.
    IntraRefresh,
}

/// Output of a single `Encoder::encode` call: an Annex-B / OBU-stream payload
/// plus the metadata transport needs.
#[derive(Debug)]
pub struct EncodedFrame {
    pub data: Vec<u8>,
    pub kind: FrameKind,
    /// Presentation timestamp in 90 kHz ticks (matches RTP convention).
    pub pts_90khz: u64,
    /// Codec parameter sets prepended to `data` for keyframes. `None` for
    /// non-keyframes. Transport may cache and re-send these on client join.
    pub parameter_sets: Option<Vec<u8>>,
}

/// A GPU-resident input frame in YUV format, owned by the caller.
///
/// Backends do not assume ownership — they record an encode command, submit
/// it, and wait. The caller is responsible for keeping the underlying VkImage
/// alive until `encode` returns.
#[derive(Debug, Clone, Copy)]
pub struct InputFrame {
    pub image: ash::vk::Image,
    pub width: u32,
    pub height: u32,
    /// Caller-supplied PTS (90 kHz ticks). Echoed back in [`EncodedFrame::pts_90khz`].
    pub pts_90khz: u64,
}

/// The contract every codec backend implements.
///
/// **Stability:** this trait is **frozen** once we ship the first backend.
/// Codec subagents (H.264, AV1) implement it without modifying it. New
/// codec-specific knobs go into codec-specific extension traits in their own
/// modules, not into this trait.
pub trait Encoder: Send {
    /// Creates a new encoder session bound to `params` on `ctx`. Validates
    /// parameters against device capabilities up front; the returned encoder
    /// is ready to accept `encode` calls immediately.
    fn new(
        ctx: std::sync::Arc<remoteway_vulkan::VulkanContext>,
        params: EncodeParams,
    ) -> Result<Self, crate::EncodeError>
    where
        Self: Sized;

    /// The Vulkan image format this encoder consumes. Caller must ensure
    /// `InputFrame::image` is in this format and in
    /// `VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR` layout.
    fn accepted_input_format(&self) -> ash::vk::Format;

    /// Encodes one frame and returns the Annex-B / OBU-stream payload.
    /// Blocks until the GPU has produced the bitstream.
    fn encode(&mut self, frame: InputFrame) -> Result<EncodedFrame, crate::EncodeError>;

    /// Forces the next encoded frame to be an IDR / keyframe, regardless of
    /// the configured intra-refresh schedule. Used by transport on packet
    /// loss recovery or client join.
    fn request_keyframe(&mut self);

    /// Reports the parameters the encoder was created with.
    fn params(&self) -> &EncodeParams;
}

#[cfg(test)]
mod tests {
    use super::*;
    use remoteway_vulkan::VideoEncodeCapabilities;

    fn caps_fixture(codec: VideoCodec) -> VideoEncodeCapabilities {
        VideoEncodeCapabilities {
            codec,
            max_coded_extent: (3840, 2160),
            min_coded_extent: (64, 64),
            picture_access_granularity: (16, 16),
            max_dpb_slots: 2,
            max_active_reference_pictures: 1,
            rate_control_modes: vk::VideoEncodeRateControlModeFlagsKHR::DISABLED
                | vk::VideoEncodeRateControlModeFlagsKHR::CBR
                | vk::VideoEncodeRateControlModeFlagsKHR::VBR,
            supports_intra_refresh: true,
            supports_quantization_map: false,
        }
    }

    fn params_fixture(codec: VideoCodec) -> EncodeParams {
        EncodeParams {
            codec,
            // 16-aligned 720p — convenient test size that satisfies typical
            // codec granularity. Note that 1080p (1920x1080) is NOT 16-aligned
            // on the height axis; callers shipping 1080p must pad to 1088 or
            // rely on backend-side padding (not yet implemented).
            width: 1280,
            height: 720,
            frame_rate: (60, 1),
            rate_control: RateControl::ConstantQp { qp: 26 },
            intra_refresh_period: None,
        }
    }

    #[test]
    fn rate_control_constant_qp_maps_to_disabled() {
        assert_eq!(
            RateControl::ConstantQp { qp: 26 }.to_vk_mode(),
            vk::VideoEncodeRateControlModeFlagsKHR::DISABLED
        );
    }

    #[test]
    fn rate_control_cbr_and_vbr_map_distinctly() {
        let cbr = RateControl::Cbr { bitrate_bps: 5_000_000 }.to_vk_mode();
        let vbr = RateControl::Vbr { average_bps: 5_000_000, peak_bps: 10_000_000 }.to_vk_mode();
        assert_ne!(cbr, vbr);
    }

    #[test]
    fn validate_happy_path() {
        let caps = caps_fixture(VideoCodec::H265);
        let params = params_fixture(VideoCodec::H265);
        params.validate_against(&caps).expect("should validate");
    }

    #[test]
    fn validate_rejects_codec_mismatch() {
        let caps = caps_fixture(VideoCodec::H265);
        let params = params_fixture(VideoCodec::H264);
        let err = params.validate_against(&caps).unwrap_err();
        assert!(matches!(err, EncodeError::InvalidParams(_)));
    }

    #[test]
    fn validate_rejects_zero_dimension() {
        let caps = caps_fixture(VideoCodec::H265);
        let mut params = params_fixture(VideoCodec::H265);
        params.width = 0;
        assert!(matches!(params.validate_against(&caps), Err(EncodeError::InvalidParams(_))));
    }

    #[test]
    fn validate_rejects_oversized_resolution() {
        let caps = caps_fixture(VideoCodec::H265);
        let mut params = params_fixture(VideoCodec::H265);
        params.width = 7680;
        params.height = 4320;
        assert!(matches!(params.validate_against(&caps), Err(EncodeError::InvalidParams(_))));
    }

    #[test]
    fn validate_rejects_misaligned_resolution() {
        let caps = caps_fixture(VideoCodec::H265);
        let mut params = params_fixture(VideoCodec::H265);
        params.width = 1281;
        assert!(matches!(params.validate_against(&caps), Err(EncodeError::InvalidParams(_))));
    }

    #[test]
    fn validate_rejects_unsupported_rate_control() {
        let mut caps = caps_fixture(VideoCodec::H265);
        // Device only supports DISABLED (CQP). Asking for CBR must fail.
        caps.rate_control_modes = vk::VideoEncodeRateControlModeFlagsKHR::DISABLED;
        let mut params = params_fixture(VideoCodec::H265);
        params.rate_control = RateControl::Cbr { bitrate_bps: 5_000_000 };
        assert!(matches!(params.validate_against(&caps), Err(EncodeError::InvalidParams(_))));
    }

    #[test]
    fn validate_rejects_intra_refresh_when_unsupported() {
        let mut caps = caps_fixture(VideoCodec::H265);
        caps.supports_intra_refresh = false;
        let mut params = params_fixture(VideoCodec::H265);
        params.intra_refresh_period = Some(30);
        assert!(matches!(params.validate_against(&caps), Err(EncodeError::InvalidParams(_))));
    }

    #[test]
    fn validate_rejects_zero_framerate_denominator() {
        let caps = caps_fixture(VideoCodec::H265);
        let mut params = params_fixture(VideoCodec::H265);
        params.frame_rate = (60, 0);
        assert!(matches!(params.validate_against(&caps), Err(EncodeError::InvalidParams(_))));
    }
}

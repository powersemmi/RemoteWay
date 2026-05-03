use crate::error::InterpolateError;
use crate::interpolator::{FrameInterpolator, GpuFrame, cpu_bicubic_upscale};

use super::fsr2_native::Fsr2NativeInterpolator;

/// FSR3 backend: Frame Generation via `FidelityFX` SDK + FSR2 upscaling.
///
/// - `interpolate()` uses `FidelityFX` Frame Generation (Vulkan).
/// - `upscale()` uses FSR 2 native via the Embark `fsr` crate.
///
/// On Linux/Wayland, requires `libFidelityFX.so` built from SDK v1.1.x
/// with Vulkan support. Set `FIDELITYFX_LIB_PATH` env var.
pub struct Fsr3Interpolator {
    /// Frame Generation backend (`FidelityFX` SDK via Vulkan).
    frame_gen: Option<super::fsr3_frame_gen::Fsr3FrameGen>,
    /// FSR 2 spatial upscaler (Vulkan).
    upscaler: Option<Fsr2NativeInterpolator>,
}

impl Fsr3Interpolator {
    /// Create a new FSR3 backend with the given render/display resolutions.
    ///
    /// Returns `InitFailed` error if the `FidelityFX` library cannot be loaded
    /// or Vulkan initialization fails.
    pub fn new(
        render_w: u32,
        render_h: u32,
        display_w: u32,
        display_h: u32,
    ) -> Result<Self, InterpolateError> {
        let frame_gen =
            super::fsr3_frame_gen::Fsr3FrameGen::new(render_w, render_h, display_w, display_h)?;
        let upscaler = Fsr2NativeInterpolator::new(display_w, display_h, display_w, display_h).ok();
        Ok(Self {
            frame_gen: Some(frame_gen),
            upscaler,
        })
    }
}

impl FrameInterpolator for Fsr3Interpolator {
    fn interpolate(
        &mut self,
        a: &GpuFrame,
        b: &GpuFrame,
        _t: f32,
    ) -> Result<GpuFrame, InterpolateError> {
        match &mut self.frame_gen {
            Some(fg) => fg.generate_frame(a, b),
            None => Err(InterpolateError::InterpolateFailed(
                "FSR3 FrameGen unavailable — is libFidelityFX.so loaded?".into(),
            )),
        }
    }

    fn upscale(
        &self,
        src: &GpuFrame,
        dst_w: u32,
        dst_h: u32,
    ) -> Result<GpuFrame, InterpolateError> {
        match &self.upscaler {
            Some(u) => u.upscale(src, dst_w, dst_h),
            None => cpu_bicubic_upscale(src, dst_w, dst_h),
        }
    }

    fn latency_ms(&self) -> f32 {
        16.67
    }

    fn name(&self) -> &str {
        "fsr3"
    }
}

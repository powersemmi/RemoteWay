//! FSR2 + RIFE backend: RIFE neural net interpolation + FSR SDK upscaling.
//!
//! `interpolate()` delegates to [`super::rife::RifeInterpolator`] for
//! neural frame interpolation. `upscale()` uses the Embark Studios `fsr`
//! crate for real AMD `FidelityFX` SDK spatial upscaling.

use crate::error::InterpolateError;
use crate::interpolator::{FrameInterpolator, GpuFrame};

/// FSR2 + RIFE hybrid backend.
pub struct Fsr2RifeInterpolator {
    /// FSR SDK for spatial upscaling.
    upscaler: super::fsr2_native::Fsr2NativeInterpolator,
}

impl Fsr2RifeInterpolator {
    /// Create a new FSR2 + RIFE hybrid.
    pub fn new() -> Result<Self, InterpolateError> {
        let upscaler = super::fsr2_native::Fsr2NativeInterpolator::new(1920, 1080, 1920, 1080)?;
        Ok(Self { upscaler })
    }
}

impl FrameInterpolator for Fsr2RifeInterpolator {
    fn interpolate(
        &mut self,
        a: &GpuFrame,
        b: &GpuFrame,
        t: f32,
    ) -> Result<GpuFrame, InterpolateError> {
        // RIFE neural network interpolation.
        let mut rife = super::rife::RifeInterpolator::from_default_path()
            .map_err(|e| InterpolateError::InterpolateFailed(format!("rife: {e}")))?;
        rife.interpolate(a, b, t)
    }

    fn latency_ms(&self) -> f32 {
        // RIFE is GPU-heavy — ~50ms per frame.
        50.0
    }

    fn upscale(
        &self,
        src: &GpuFrame,
        dst_w: u32,
        dst_h: u32,
    ) -> Result<GpuFrame, InterpolateError> {
        self.upscaler.upscale(src, dst_w, dst_h)
    }

    fn name(&self) -> &'static str {
        "fsr2-rife"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fsr2_rife_name() {
        assert!(!Fsr2RifeInterpolator.name().is_empty());
    }
}

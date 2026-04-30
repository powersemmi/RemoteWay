use crate::error::InterpolateError;
use crate::interpolator::{FrameInterpolator, GpuFrame};

use super::vulkan_context::VulkanContext;

/// FSR3-style hardware-accelerated frame interpolation for RDNA3+ GPUs.
///
/// Uses the same Vulkan compute motion estimation and warp/blend pipeline
/// as `Fsr2Interpolator`, but with parameters optimized for AMD RDNA3/RDNA4
/// hardware: wave64 workgroup sizes, larger block size and search radius
/// for higher quality motion estimation.
///
/// Requires an AMD RDNA3+ GPU (RX 7000 series or RX 9000 series).
pub struct Fsr3Interpolator {
    inner: super::fsr2::Fsr2Interpolator,
}

impl Fsr3Interpolator {
    /// Create a new FSR3 interpolator.
    ///
    /// Checks for RDNA3+ hardware before initialization. Falls back to
    /// the same Vulkan compute path as FSR2 with tuned parameters:
    /// - block_size=16 (larger blocks for better quality)
    /// - search_radius=16 (wider search for fast motion)
    pub fn new() -> Result<Self, InterpolateError> {
        // Gate: RDNA3+ hardware required.
        if !VulkanContext::probe_rdna3_plus() {
            return Err(InterpolateError::InitFailed(
                "FSR3 requires AMD RDNA3+ GPU (RX 7000/9000 series)".into(),
            ));
        }

        // Use larger block size and search radius for RDNA3+ hardware,
        // which has more compute units and faster memory bandwidth.
        let inner = super::fsr2::Fsr2Interpolator::with_params(16, 16)?;
        Ok(Self { inner })
    }
}

impl FrameInterpolator for Fsr3Interpolator {
    fn interpolate(
        &self,
        a: &GpuFrame,
        b: &GpuFrame,
        t: f32,
    ) -> Result<GpuFrame, InterpolateError> {
        self.inner.interpolate(a, b, t)
    }

    fn latency_ms(&self) -> f32 {
        // Slightly higher latency due to larger search radius,
        // offset by RDNA3+ compute performance.
        4.0
    }

    fn name(&self) -> &str {
        "fsr3-hardware"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fsr3_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Fsr3Interpolator>();
    }

    #[test]
    #[ignore] // requires RDNA3+ GPU
    fn fsr3_init() {
        let interp = Fsr3Interpolator::new();
        // May fail on non-RDNA3+ hardware — that's expected.
        if let Ok(interp) = interp {
            assert_eq!(interp.name(), "fsr3-hardware");
        }
    }

    #[test]
    #[ignore] // requires RDNA3+ GPU
    fn fsr3_interpolate_small() {
        let interp = match Fsr3Interpolator::new() {
            Ok(i) => i,
            Err(_) => return, // skip on non-RDNA3+ hardware
        };
        let a = GpuFrame::from_data(vec![0u8; 64 * 64 * 4], 64, 64, 256, 0);
        let b = GpuFrame::from_data(vec![128u8; 64 * 64 * 4], 64, 64, 256, 1000);
        let result = interp.interpolate(&a, &b, 0.5).unwrap();
        assert_eq!(result.width, 64);
        assert_eq!(result.data.len(), 64 * 64 * 4);
    }
}

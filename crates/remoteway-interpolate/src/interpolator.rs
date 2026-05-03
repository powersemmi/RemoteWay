//! Core frame interpolation traits and CPU fallback.
//!
//! Defines [`FrameInterpolator`], [`GpuFrame`], and the universal
//! [`LinearBlendInterpolator`] CPU fallback.

use crate::error::InterpolateError;

/// A frame in CPU memory ready for interpolation.
///
/// Contains raw RGBA pixel data and metadata. This is the interchange
/// format between pipeline stages — GPU backends copy data to/from
/// device memory internally.
#[derive(Debug)]
#[must_use]
pub struct GpuFrame {
    /// Raw pixel data in BGRA/XRGB 8888 format.
    pub data: Vec<u8>,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Bytes per row (may include padding).
    pub stride: u32,
    /// Capture timestamp in nanoseconds.
    pub timestamp_ns: u64,
}

impl GpuFrame {
    /// Create a new frame with the given dimensions.
    ///
    /// Allocates `stride * height` bytes zeroed.
    pub fn new(width: u32, height: u32, stride: u32, timestamp_ns: u64) -> Self {
        Self {
            data: vec![0u8; (stride * height) as usize],
            width,
            height,
            stride,
            timestamp_ns,
        }
    }

    /// Create a frame from existing pixel data.
    pub fn from_data(
        data: Vec<u8>,
        width: u32,
        height: u32,
        stride: u32,
        timestamp_ns: u64,
    ) -> Self {
        Self {
            data,
            width,
            height,
            stride,
            timestamp_ns,
        }
    }

    /// Total size in bytes of the pixel data.
    #[must_use]
    pub fn byte_size(&self) -> usize {
        self.data.len()
    }

    /// Number of pixels (width × height).
    #[must_use]
    pub fn pixel_count(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Check if this frame has the same dimensions as another.
    #[must_use]
    pub fn same_dimensions(&self, other: &GpuFrame) -> bool {
        self.width == other.width && self.height == other.height && self.stride == other.stride
    }
}

/// Core trait for frame interpolation backends.
///
/// Implementations generate intermediate frames between two real captured
/// frames using motion estimation (optical flow) and warping. The `t`
/// parameter controls temporal position: 0.0 = frame `a`, 1.0 = frame `b`.
pub trait FrameInterpolator: Send + Sync {
    /// Interpolate between frames `a` and `b` at temporal position `t`.
    ///
    /// `t` must be in the range `0.0..=1.0`. Returns a new synthesized frame.
    ///
    /// # Errors
    ///
    /// Returns [`InterpolateError::InvalidFactor`] if `t` is outside `0.0..=1.0`,
    /// [`InterpolateError::DimensionMismatch`] if frames differ in size, or
    /// [`InterpolateError::InterpolateFailed`] on GPU/compute errors.
    fn interpolate(&mut self, a: &GpuFrame, b: &GpuFrame, t: f32)
    -> Result<GpuFrame, InterpolateError>;

    /// Spatially upscale a single frame to `dst_w×dst_h`.
    ///
    /// Default implementation uses CPU Catmull-Rom bicubic. GPU backends
    /// (FSR2, FSR3, DLSS) override this with hardware-accelerated upscaling.
    ///
    /// # Errors
    ///
    /// Returns [`InterpolateError::InterpolateFailed`] on GPU/compute errors.
    fn upscale(
        &self,
        src: &GpuFrame,
        dst_w: u32,
        dst_h: u32,
    ) -> Result<GpuFrame, InterpolateError> {
        cpu_bicubic_upscale(src, dst_w, dst_h)
    }

    /// Estimated latency of a single interpolation call in milliseconds.
    fn latency_ms(&self) -> f32;

    /// Human-readable name of this backend.
    fn name(&self) -> &str;
}

/// CPU Catmull-Rom bicubic upscale.
///
/// Standalone implementation used as default for `FrameInterpolator::upscale()`
/// and as fallback when GPU upscaling is unavailable.
pub fn cpu_bicubic_upscale(
    src: &GpuFrame,
    dst_w: u32,
    dst_h: u32,
) -> Result<GpuFrame, InterpolateError> {
    let dst_stride = dst_w * 4;
    let mut data = vec![0u8; (dst_stride * dst_h) as usize];

    fn cubic_weight(t: f64) -> [f64; 4] {
        let t2 = t * t;
        let t3 = t2 * t;
        [
            0.5 * (-t3 + 2.0 * t2 - t),
            0.5 * (3.0 * t3 - 5.0 * t2 + 2.0),
            0.5 * (-3.0 * t3 + 4.0 * t2 + t),
            0.5 * (t3 - t2),
        ]
    }

    for dy in 0..dst_h {
        let sy = (dy as f64 + 0.5) * src.height as f64 / dst_h as f64 - 0.5;
        let sy_floor = (sy.floor() as i32).max(0).min(src.height as i32 - 1);
        let fy = sy - sy.floor();
        let sy0 = (sy_floor - 1).max(0) as u32;
        let sy1 = sy_floor as u32;
        let sy2 = (sy_floor + 1).min(src.height as i32 - 1) as u32;
        let sy3 = (sy_floor + 2).min(src.height as i32 - 1) as u32;
        let wy = cubic_weight(fy);

        let dst_row = (dy * dst_stride) as usize;
        for dx in 0..dst_w {
            let sx = (dx as f64 + 0.5) * src.width as f64 / dst_w as f64 - 0.5;
            let sx_floor = (sx.floor() as i32).max(0).min(src.width as i32 - 1);
            let fx = sx - sx.floor();
            let sx0 = (sx_floor - 1).max(0) as u32;
            let sx1 = sx_floor as u32;
            let sx2 = (sx_floor + 1).min(src.width as i32 - 1) as u32;
            let sx3 = (sx_floor + 2).min(src.width as i32 - 1) as u32;
            let wx = cubic_weight(fx);

            let di = dst_row + (dx * 4) as usize;
            for c in 0..4 {
                let mut acc = 0.0f64;
                for (ky, &row_idx) in [sy0, sy1, sy2, sy3].iter().enumerate() {
                    let row = (row_idx * src.stride) as usize;
                    let cols = [sx0, sx1, sx2, sx3];
                    for (kx, &col_idx) in cols.iter().enumerate() {
                        let si = row + (col_idx * 4) as usize + c;
                        acc += f64::from(src.data[si]) * wx[kx] * wy[ky];
                    }
                }
                data[di + c] = acc.round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    Ok(GpuFrame {
        data,
        width: dst_w,
        height: dst_h,
        stride: dst_stride,
        timestamp_ns: src.timestamp_ns,
    })
}

/// CPU-only linear blend interpolation (universal fallback).
///
/// Performs per-pixel alpha blending between frames. No motion estimation —
/// fast but produces ghosting on moving objects. Suitable as a baseline
/// and for static/slow content.
pub struct LinearBlendInterpolator;

impl FrameInterpolator for LinearBlendInterpolator {
    fn interpolate(
        &mut self,
        a: &GpuFrame,
        b: &GpuFrame,
        t: f32,
    ) -> Result<GpuFrame, InterpolateError> {
        if !(0.0..=1.0).contains(&t) {
            return Err(InterpolateError::InvalidFactor(t));
        }
        if !a.same_dimensions(b) {
            return Err(InterpolateError::DimensionMismatch(
                a.width, a.height, b.width, b.height,
            ));
        }

        let len = a.data.len().min(b.data.len());
        let mut result = Vec::with_capacity(len);

        // Quantize blend factor to 0..256 for integer math (no FP on hot path).
        // SAFETY: t is validated to be in 0.0..=1.0, so t*256.0 is in 0.0..=256.0.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let t_fixed = (t * 256.0) as u16;
        let inv_t = 256 - t_fixed;

        for i in 0..len {
            let va = u16::from(a.data[i]);
            let vb = u16::from(b.data[i]);
            let blended = (va * inv_t + vb * t_fixed) >> 8;
            result.push(blended as u8);
        }

        // Interpolated timestamp: linear between a and b.
        // NOTE: delta (u64) to f64 conversion loses precision for very large
        // values (>2^53 ns ~ 104 days), which is irrelevant for frame timing.
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let ts = if b.timestamp_ns >= a.timestamp_ns {
            let delta = b.timestamp_ns - a.timestamp_ns;
            a.timestamp_ns + (delta as f64 * f64::from(t)) as u64
        } else {
            a.timestamp_ns
        };

        Ok(GpuFrame {
            data: result,
            width: a.width,
            height: a.height,
            stride: a.stride,
            timestamp_ns: ts,
        })
    }

    fn latency_ms(&self) -> f32 {
        // ~0.5ms for 1080p on modern CPU.
        0.5
    }

    fn name(&self) -> &'static str {
        "linear-blend"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(width: u32, height: u32, value: u8, ts: u64) -> GpuFrame {
        let stride = width * 4;
        GpuFrame {
            data: vec![value; (stride * height) as usize],
            width,
            height,
            stride,
            timestamp_ns: ts,
        }
    }

    #[test]
    fn gpu_frame_new() {
        let f = GpuFrame::new(1920, 1080, 1920 * 4, 0);
        assert_eq!(f.width, 1920);
        assert_eq!(f.height, 1080);
        assert_eq!(f.stride, 7680);
        assert_eq!(f.byte_size(), 1920 * 1080 * 4);
        assert_eq!(f.pixel_count(), 1920 * 1080);
    }

    #[test]
    fn gpu_frame_from_data() {
        let data = vec![0xFF; 64 * 48 * 4];
        let f = GpuFrame::from_data(data, 64, 48, 256, 12345);
        assert_eq!(f.byte_size(), 64 * 48 * 4);
        assert_eq!(f.timestamp_ns, 12345);
    }

    #[test]
    fn gpu_frame_same_dimensions() {
        let a = GpuFrame::new(100, 50, 400, 0);
        let b = GpuFrame::new(100, 50, 400, 1);
        let c = GpuFrame::new(200, 50, 800, 0);
        assert!(a.same_dimensions(&b));
        assert!(!a.same_dimensions(&c));
    }

    #[test]
    fn linear_blend_at_zero() {
        let mut interp = LinearBlendInterpolator;
        let a = make_frame(2, 2, 100, 0);
        let b = make_frame(2, 2, 200, 1000);
        let result = interp.interpolate(&a, &b, 0.0).unwrap();
        // At t=0, result should be very close to frame a.
        for &px in &result.data {
            assert!((99..=101).contains(&px), "px={px}");
        }
        assert_eq!(result.timestamp_ns, 0);
    }

    #[test]
    fn linear_blend_at_one() {
        let mut interp = LinearBlendInterpolator;
        let a = make_frame(2, 2, 100, 0);
        let b = make_frame(2, 2, 200, 1000);
        let result = interp.interpolate(&a, &b, 1.0).unwrap();
        for &px in &result.data {
            assert!((199..=201).contains(&px), "px={px}");
        }
        assert_eq!(result.timestamp_ns, 1000);
    }

    #[test]
    fn linear_blend_at_half() {
        let mut interp = LinearBlendInterpolator;
        let a = make_frame(2, 2, 0, 0);
        let b = make_frame(2, 2, 200, 2000);
        let result = interp.interpolate(&a, &b, 0.5).unwrap();
        for &px in &result.data {
            // 0 * 128/256 + 200 * 128/256 = 100
            assert!((98..=102).contains(&px), "px={px}");
        }
        assert_eq!(result.timestamp_ns, 1000);
    }

    #[test]
    fn linear_blend_black_white() {
        let mut interp = LinearBlendInterpolator;
        let a = make_frame(4, 4, 0, 0);
        let b = make_frame(4, 4, 255, 16_666_667);
        let result = interp.interpolate(&a, &b, 0.25).unwrap();
        // 0 * 192/256 + 255 * 64/256 = 63.75 ≈ 63 or 64
        for &px in &result.data {
            assert!((62..=65).contains(&px), "px={px}");
        }
    }

    #[test]
    fn linear_blend_dimension_mismatch() {
        let mut interp = LinearBlendInterpolator;
        let a = make_frame(100, 100, 128, 0);
        let b = make_frame(200, 100, 128, 1000);
        let result = interp.interpolate(&a, &b, 0.5);
        assert!(matches!(
            result,
            Err(InterpolateError::DimensionMismatch(..))
        ));
    }

    #[test]
    fn linear_blend_invalid_factor() {
        let mut interp = LinearBlendInterpolator;
        let a = make_frame(2, 2, 128, 0);
        let b = make_frame(2, 2, 128, 1000);

        assert!(matches!(
            interp.interpolate(&a, &b, -0.1),
            Err(InterpolateError::InvalidFactor(_))
        ));
        assert!(matches!(
            interp.interpolate(&a, &b, 1.1),
            Err(InterpolateError::InvalidFactor(_))
        ));
    }

    #[test]
    fn linear_blend_name_and_latency() {
        let mut interp = LinearBlendInterpolator;
        assert_eq!(interp.name(), "linear-blend");
        assert!(interp.latency_ms() > 0.0);
    }

    #[test]
    fn frame_interpolator_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<LinearBlendInterpolator>();
    }

    #[test]
    fn linear_blend_1080p() {
        let mut interp = LinearBlendInterpolator;
        let a = make_frame(1920, 1080, 128, 0);
        let b = make_frame(1920, 1080, 64, 16_666_667);
        let result = interp.interpolate(&a, &b, 0.5).unwrap();
        assert_eq!(result.width, 1920);
        assert_eq!(result.height, 1080);
        assert_eq!(result.byte_size(), 1920 * 1080 * 4);
    }
}

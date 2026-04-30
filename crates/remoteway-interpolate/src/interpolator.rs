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
    pub fn byte_size(&self) -> usize {
        self.data.len()
    }

    /// Number of pixels (width × height).
    pub fn pixel_count(&self) -> u64 {
        self.width as u64 * self.height as u64
    }

    /// Check if this frame has the same dimensions as another.
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
    fn interpolate(&self, a: &GpuFrame, b: &GpuFrame, t: f32)
    -> Result<GpuFrame, InterpolateError>;

    /// Estimated latency of a single interpolation call in milliseconds.
    fn latency_ms(&self) -> f32;

    /// Human-readable name of this backend.
    fn name(&self) -> &str;
}

/// CPU-only linear blend interpolation (universal fallback).
///
/// Performs per-pixel alpha blending between frames. No motion estimation —
/// fast but produces ghosting on moving objects. Suitable as a baseline
/// and for static/slow content.
pub struct LinearBlendInterpolator;

impl FrameInterpolator for LinearBlendInterpolator {
    fn interpolate(
        &self,
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
        let t_fixed = (t * 256.0) as u16;
        let inv_t = 256 - t_fixed;

        for i in 0..len {
            let va = a.data[i] as u16;
            let vb = b.data[i] as u16;
            let blended = (va * inv_t + vb * t_fixed) >> 8;
            result.push(blended as u8);
        }

        // Interpolated timestamp: linear between a and b.
        let ts = if b.timestamp_ns >= a.timestamp_ns {
            let delta = b.timestamp_ns - a.timestamp_ns;
            a.timestamp_ns + (delta as f64 * t as f64) as u64
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

    fn name(&self) -> &str {
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
        let interp = LinearBlendInterpolator;
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
        let interp = LinearBlendInterpolator;
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
        let interp = LinearBlendInterpolator;
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
        let interp = LinearBlendInterpolator;
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
        let interp = LinearBlendInterpolator;
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
        let interp = LinearBlendInterpolator;
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
        let interp = LinearBlendInterpolator;
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
        let interp = LinearBlendInterpolator;
        let a = make_frame(1920, 1080, 128, 0);
        let b = make_frame(1920, 1080, 64, 16_666_667);
        let result = interp.interpolate(&a, &b, 0.5).unwrap();
        assert_eq!(result.width, 1920);
        assert_eq!(result.height, 1080);
        assert_eq!(result.byte_size(), 1920 * 1080 * 4);
    }
}

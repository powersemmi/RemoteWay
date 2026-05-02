use crate::error::InterpolateError;
use crate::interpolator::{FrameInterpolator, GpuFrame};

/// Manages frame buffering and interpolation between real captured frames.
///
/// Maintains a sliding window of the two most recent frames (anchor and target)
/// and generates interpolated frames between them on demand. Handles anchor
/// frame transitions and error accumulation reset.
pub struct InterpolationManager {
    backend: Box<dyn FrameInterpolator>,
    /// Previous frame (anchor for interpolation).
    anchor: Option<GpuFrame>,
    /// Latest received frame (target for interpolation).
    target: Option<GpuFrame>,
    /// Number of real frames received.
    frame_count: u64,
    /// Number of interpolated frames generated.
    interpolated_count: u64,
    /// Force anchor reset on next frame (e.g., after scene change).
    force_reset: bool,
}

impl InterpolationManager {
    /// Create a new manager with the given interpolation backend.
    #[must_use]
    pub fn new(backend: Box<dyn FrameInterpolator>) -> Self {
        Self {
            backend,
            anchor: None,
            target: None,
            frame_count: 0,
            interpolated_count: 0,
            force_reset: false,
        }
    }

    /// Push a new real frame into the manager.
    ///
    /// The previous target becomes the anchor, and this frame becomes
    /// the new target. If `force_reset` is set, the anchor is cleared
    /// to avoid interpolating across a scene change.
    pub fn push_frame(&mut self, frame: GpuFrame) {
        if self.force_reset {
            self.anchor = None;
            self.force_reset = false;
        } else {
            self.anchor = self.target.take();
        }
        self.target = Some(frame);
        self.frame_count += 1;
    }

    /// Generate an interpolated frame at temporal position `t` between
    /// the anchor and target frames.
    ///
    /// Returns `None` if fewer than 2 frames have been pushed (no pair to
    /// interpolate between).
    ///
    /// # Errors
    ///
    /// Propagates errors from the underlying [`FrameInterpolator::interpolate`]
    /// call (invalid factor, dimension mismatch, GPU/compute errors).
    pub fn interpolate(&mut self, t: f32) -> Result<Option<GpuFrame>, InterpolateError> {
        let (Some(anchor), Some(target)) = (&self.anchor, &self.target) else {
            return Ok(None);
        };

        let result = self.backend.interpolate(anchor, target, t)?;
        self.interpolated_count += 1;
        Ok(Some(result))
    }

    /// Check if the manager has a valid frame pair for interpolation.
    #[must_use]
    pub fn can_interpolate(&self) -> bool {
        self.anchor.is_some() && self.target.is_some()
    }

    /// Force an anchor reset on the next `push_frame`.
    ///
    /// Use this after scene changes, seek events, or when the server
    /// sends an anchor/keyframe that breaks continuity with previous frames.
    pub fn reset_anchor(&mut self) {
        self.force_reset = true;
    }

    /// Clear all state — both frames and counters.
    pub fn clear(&mut self) {
        self.anchor = None;
        self.target = None;
        self.frame_count = 0;
        self.interpolated_count = 0;
        self.force_reset = false;
    }

    /// Number of real frames received.
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Number of interpolated frames generated.
    #[must_use]
    pub fn interpolated_count(&self) -> u64 {
        self.interpolated_count
    }

    /// Name of the active interpolation backend.
    #[must_use]
    pub fn backend_name(&self) -> &str {
        self.backend.name()
    }

    /// Estimated latency of the active backend in milliseconds.
    #[must_use]
    pub fn backend_latency_ms(&self) -> f32 {
        self.backend.latency_ms()
    }

    /// Timestamp of the anchor frame, if available.
    #[must_use]
    pub fn anchor_timestamp_ns(&self) -> Option<u64> {
        self.anchor.as_ref().map(|f| f.timestamp_ns)
    }

    /// Timestamp of the target frame, if available.
    #[must_use]
    pub fn target_timestamp_ns(&self) -> Option<u64> {
        self.target.as_ref().map(|f| f.timestamp_ns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpolator::LinearBlendInterpolator;

    fn make_frame(value: u8, ts: u64) -> GpuFrame {
        GpuFrame {
            data: vec![value; 64 * 48 * 4],
            width: 64,
            height: 48,
            stride: 256,
            timestamp_ns: ts,
        }
    }

    fn make_manager() -> InterpolationManager {
        InterpolationManager::new(Box::new(LinearBlendInterpolator))
    }

    #[test]
    fn new_manager_empty() {
        let mgr = make_manager();
        assert!(!mgr.can_interpolate());
        assert_eq!(mgr.frame_count(), 0);
        assert_eq!(mgr.interpolated_count(), 0);
        assert_eq!(mgr.backend_name(), "linear-blend");
        assert!(mgr.anchor_timestamp_ns().is_none());
        assert!(mgr.target_timestamp_ns().is_none());
    }

    #[test]
    fn single_frame_cannot_interpolate() {
        let mut mgr = make_manager();
        mgr.push_frame(make_frame(100, 0));
        assert!(!mgr.can_interpolate());
        assert_eq!(mgr.frame_count(), 1);
        let result = mgr.interpolate(0.5).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn two_frames_can_interpolate() {
        let mut mgr = make_manager();
        mgr.push_frame(make_frame(0, 0));
        mgr.push_frame(make_frame(200, 16_666_667));
        assert!(mgr.can_interpolate());
        assert_eq!(mgr.frame_count(), 2);
        assert_eq!(mgr.anchor_timestamp_ns(), Some(0));
        assert_eq!(mgr.target_timestamp_ns(), Some(16_666_667));

        let result = mgr.interpolate(0.5).unwrap().unwrap();
        assert_eq!(result.width, 64);
        assert_eq!(result.height, 48);
        assert_eq!(mgr.interpolated_count(), 1);
        // Blended value should be ~100
        for &px in &result.data {
            assert!((98..=102).contains(&px), "px={px}");
        }
    }

    #[test]
    fn push_slides_window() {
        let mut mgr = make_manager();
        mgr.push_frame(make_frame(0, 0));
        mgr.push_frame(make_frame(100, 1000));
        mgr.push_frame(make_frame(200, 2000));
        // anchor should now be frame with value 100
        assert_eq!(mgr.anchor_timestamp_ns(), Some(1000));
        assert_eq!(mgr.target_timestamp_ns(), Some(2000));
        assert_eq!(mgr.frame_count(), 3);
    }

    #[test]
    fn reset_anchor_clears_on_next_push() {
        let mut mgr = make_manager();
        mgr.push_frame(make_frame(0, 0));
        mgr.push_frame(make_frame(100, 1000));
        assert!(mgr.can_interpolate());

        mgr.reset_anchor();
        mgr.push_frame(make_frame(200, 2000));
        // After reset, anchor is None — only target exists.
        assert!(!mgr.can_interpolate());
        assert!(mgr.anchor_timestamp_ns().is_none());
        assert_eq!(mgr.target_timestamp_ns(), Some(2000));
    }

    #[test]
    fn clear_resets_everything() {
        let mut mgr = make_manager();
        mgr.push_frame(make_frame(0, 0));
        mgr.push_frame(make_frame(100, 1000));
        let _ = mgr.interpolate(0.5).unwrap();

        mgr.clear();
        assert!(!mgr.can_interpolate());
        assert_eq!(mgr.frame_count(), 0);
        assert_eq!(mgr.interpolated_count(), 0);
    }

    #[test]
    fn backend_latency() {
        let mgr = make_manager();
        assert!(mgr.backend_latency_ms() > 0.0);
    }

    #[test]
    fn multiple_interpolations_between_same_pair() {
        let mut mgr = make_manager();
        mgr.push_frame(make_frame(0, 0));
        mgr.push_frame(make_frame(255, 10000));

        for i in 1..=5 {
            let t = i as f32 / 6.0;
            let result = mgr.interpolate(t).unwrap().unwrap();
            assert_eq!(result.width, 64);
        }
        assert_eq!(mgr.interpolated_count(), 5);
    }
}

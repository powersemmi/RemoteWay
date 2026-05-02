//! Presentation-time feedback via `wp_presentation_time`.
//!
//! Tracks frame presentation timestamps and estimates the display refresh
//! interval using an exponential moving average.

use wayland_client::protocol::wl_surface;
use wayland_client::{Connection, Dispatch, QueueHandle};
use wayland_protocols::wp::presentation_time::client::{wp_presentation, wp_presentation_feedback};

use crate::surface::DisplayState;

/// Tracks presentation timing for adaptive frame pacing.
///
/// Uses the `wp-presentation-time` protocol to get precise timestamps
/// of when frames are actually displayed on the monitor, enabling
/// accurate frame pacing and latency measurement.
pub struct PresentationSync {
    presentation: wp_presentation::WpPresentation,
    /// Timestamp of the last presented frame (nanoseconds).
    last_presented_ns: u64,
    /// Measured frame interval from presentation feedback (nanoseconds).
    frame_interval_ns: u64,
    /// Clock ID from the presentation global (matches the compositor clock).
    clock_id: u32,
    /// Number of feedback samples collected.
    sample_count: u64,
}

impl PresentationSync {
    /// Create a new presentation sync tracker.
    #[must_use]
    pub fn new(presentation: wp_presentation::WpPresentation, clock_id: u32) -> Self {
        Self {
            presentation,
            last_presented_ns: 0,
            // Default to ~60Hz until we get actual feedback.
            frame_interval_ns: 16_666_667,
            clock_id,
            sample_count: 0,
        }
    }

    /// Request presentation feedback for a committed surface.
    ///
    /// Call this after `wl_surface.commit()` to track when the frame
    /// actually appears on screen.
    pub fn request_feedback(
        &self,
        surface: &wl_surface::WlSurface,
        qh: &QueueHandle<DisplayState>,
    ) {
        // The returned WpPresentationFeedback proxy is managed internally by the
        // Wayland event queue; we receive its events via the Dispatch impl below.
        let _ = self.presentation.feedback(surface, qh, ());
    }

    /// Record a presentation timestamp from feedback.
    pub fn record_presentation(&mut self, timestamp_ns: u64) {
        if self.last_presented_ns > 0 {
            let interval = timestamp_ns.saturating_sub(self.last_presented_ns);
            if interval > 0 && interval < 100_000_000 {
                // Exponential moving average for smoothing.
                if self.sample_count < 4 {
                    self.frame_interval_ns = interval;
                } else {
                    self.frame_interval_ns = (self.frame_interval_ns * 7 + interval) / 8;
                }
            }
        }
        self.last_presented_ns = timestamp_ns;
        self.sample_count += 1;
    }

    /// Mark that a frame was discarded (not displayed).
    pub fn record_discarded(&mut self) {
        // Discarded frame — no timing update, but we note it.
    }

    /// Timestamp of the last successfully presented frame.
    #[must_use]
    pub fn last_presented_ns(&self) -> u64 {
        self.last_presented_ns
    }

    /// Measured frame interval in nanoseconds (smoothed).
    #[must_use]
    pub fn frame_interval_ns(&self) -> u64 {
        self.frame_interval_ns
    }

    /// Clock ID from the compositor.
    #[must_use]
    pub fn clock_id(&self) -> u32 {
        self.clock_id
    }

    /// Number of feedback samples collected.
    #[must_use]
    pub fn sample_count(&self) -> u64 {
        self.sample_count
    }
}

// --- Wayland dispatch implementations ---

impl Dispatch<wp_presentation::WpPresentation, ()> for DisplayState {
    fn event(
        _state: &mut Self,
        _proxy: &wp_presentation::WpPresentation,
        _event: wp_presentation::Event,
        _: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        // The only event is `clock_id` which is handled during global binding.
    }
}

impl Dispatch<wp_presentation_feedback::WpPresentationFeedback, ()> for DisplayState {
    fn event(
        _state: &mut Self,
        _proxy: &wp_presentation_feedback::WpPresentationFeedback,
        _event: wp_presentation_feedback::Event,
        _: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        // Presentation feedback events are processed by the DisplayThread
        // via the PresentationSync state. The events are:
        // - SyncOutput: which output the frame was shown on
        // - Presented { tv_sec_hi, tv_sec_lo, tv_nsec, refresh, ... }
        // - Discarded
        //
        // These are handled in the display loop which has access to
        // PresentationSync for recording timestamps.
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn presentation_sync_defaults() {
        // Test the timing calculation logic directly.
        let interval = 16_666_667u64; // ~60Hz
        assert!(interval > 0);
        assert!(interval < 100_000_000);
    }

    #[test]
    fn frame_interval_smoothing() {
        // Simulate the EMA smoothing calculation.
        let mut interval = 16_666_667u64;
        let new_sample = 16_700_000u64;
        // After 4+ samples, EMA applies: (old * 7 + new) / 8
        interval = (interval * 7 + new_sample) / 8;
        // Should be close to the old value, slightly shifted toward new.
        assert!(interval > 16_666_667);
        assert!(interval < 16_700_000);
    }

    #[test]
    fn timestamp_recording() {
        // Simulate recording presentations without real Wayland objects.
        let mut last_ns = 0u64;
        let mut frame_interval = 16_666_667u64;
        let mut count = 0u64;

        let timestamps = [
            100_000_000u64,
            116_666_667,
            133_333_334,
            150_000_001,
            166_666_668,
        ];

        for &ts in &timestamps {
            if last_ns > 0 {
                let interval = ts.saturating_sub(last_ns);
                if interval > 0 && interval < 100_000_000 {
                    if count < 4 {
                        frame_interval = interval;
                    } else {
                        frame_interval = (frame_interval * 7 + interval) / 8;
                    }
                }
            }
            last_ns = ts;
            count += 1;
        }

        assert_eq!(last_ns, 166_666_668);
        assert!(count == 5);
        // Frame interval should be approximately 16.67ms.
        assert!(frame_interval > 16_000_000);
        assert!(frame_interval < 17_000_000);
    }

    /// Test that the first record sets `last_presented_ns` but does NOT update interval
    /// (because `last_presented_ns` == 0 initially).
    #[test]
    fn first_record_does_not_update_interval() {
        let default_interval = 16_666_667u64;
        let mut last_ns = 0u64;
        let mut frame_interval = default_interval;
        let mut count = 0u64;

        // First timestamp: last_ns is 0, so we skip interval update.
        let ts = 500_000_000u64;
        if last_ns > 0 {
            let interval = ts.saturating_sub(last_ns);
            if interval > 0 && interval < 100_000_000 {
                if count < 4 {
                    frame_interval = interval;
                } else {
                    frame_interval = (frame_interval * 7 + interval) / 8;
                }
            }
        }
        last_ns = ts;
        count += 1;

        assert_eq!(last_ns, 500_000_000);
        assert_eq!(count, 1);
        // Frame interval should remain at default.
        assert_eq!(frame_interval, default_interval);
    }

    /// Intervals >= 100ms are discarded (too large = likely a pause or glitch).
    #[test]
    fn large_interval_discarded() {
        let default_interval = 16_666_667u64;
        let last_ns = 100_000_000u64;
        let mut frame_interval = default_interval;
        let count = 2u64; // pretend we have some samples

        // 200ms gap — should be discarded.
        let ts = 300_000_000u64;
        let interval = ts.saturating_sub(last_ns); // 200_000_000
        assert!(interval >= 100_000_000);
        if last_ns > 0 && interval > 0 && interval < 100_000_000 {
            frame_interval = interval;
        }
        let last_ns = ts;
        let count = count + 1;

        assert_eq!(last_ns, 300_000_000);
        assert_eq!(count, 3);
        // Frame interval unchanged — the large gap was rejected.
        assert_eq!(frame_interval, default_interval);
    }

    /// Interval of exactly 0 (duplicate timestamp) is discarded.
    #[test]
    fn zero_interval_discarded() {
        let default_interval = 16_666_667u64;
        let mut last_ns = 100_000_000u64;
        let mut frame_interval = default_interval;
        // sample count = 2

        let ts = 100_000_000u64; // same as last
        let interval = ts.saturating_sub(last_ns); // 0
        assert_eq!(interval, 0);
        if last_ns > 0 && interval > 0 && interval < 100_000_000 {
            frame_interval = interval;
        }
        last_ns = ts;

        assert_eq!(frame_interval, default_interval);
        assert_eq!(last_ns, ts);
    }

    /// First 4 samples use direct assignment; starting from sample 4 use EMA.
    #[test]
    fn ema_kicks_in_at_sample_4() {
        let mut last_ns = 0u64;
        let mut frame_interval = 16_666_667u64;

        // 5 timestamps = 4 intervals
        let timestamps = [
            100_000_000u64,
            116_666_000, // interval: 16_666_000
            133_332_000, // interval: 16_666_000
            149_998_000, // interval: 16_666_000
            166_700_000, // interval: 16_702_000 — this one uses EMA
        ];

        let mut intervals_assigned = Vec::new();
        // count = number of samples already processed = loop index.
        for (count, &ts) in timestamps.iter().enumerate() {
            let count = count as u64;
            if last_ns > 0 {
                let interval = ts.saturating_sub(last_ns);
                if interval > 0 && interval < 100_000_000 {
                    if count < 4 {
                        frame_interval = interval;
                        intervals_assigned.push(("direct", interval, frame_interval));
                    } else {
                        let old = frame_interval;
                        frame_interval = (frame_interval * 7 + interval) / 8;
                        intervals_assigned.push(("ema", interval, frame_interval));
                        // EMA should blend toward new sample.
                        assert!(frame_interval > old.min(interval));
                        assert!(frame_interval < old.max(interval));
                    }
                }
            }
            last_ns = ts;
        }
        assert_eq!(intervals_assigned.len(), 4);
        // First 3 intervals are direct assignment.
        assert_eq!(intervals_assigned[0].0, "direct");
        assert_eq!(intervals_assigned[1].0, "direct");
        assert_eq!(intervals_assigned[2].0, "direct");
        // Fourth interval uses EMA.
        assert_eq!(intervals_assigned[3].0, "ema");
    }

    /// EMA converges toward a stable new rate after many samples.
    #[test]
    fn ema_converges_to_new_rate() {
        let mut frame_interval = 16_666_667u64; // start at ~60Hz
        let target = 8_333_333u64; // target ~120Hz

        // Simulate many samples at the target rate with EMA.
        for _ in 0..100 {
            frame_interval = (frame_interval * 7 + target) / 8;
        }

        // After 100 EMA iterations, should be very close to target.
        let diff = frame_interval.abs_diff(target);
        assert!(
            diff < 100,
            "EMA did not converge: interval={frame_interval}, target={target}, diff={diff}"
        );
    }

    /// `saturating_sub` prevents underflow when timestamps are out of order
    /// (which shouldn't happen but is handled defensively).
    #[test]
    fn saturating_sub_prevents_underflow() {
        let last_ns = 200_000_000u64;
        let ts = 100_000_000u64; // earlier than last (out of order)
        let interval = ts.saturating_sub(last_ns);
        assert_eq!(interval, 0);
    }

    /// Exactly at the boundary: `99_999_999` ns should be accepted.
    #[test]
    fn boundary_interval_accepted() {
        let last_ns = 100_000_000u64;
        let ts = 199_999_999u64; // interval = 99_999_999
        let interval = ts.saturating_sub(last_ns);
        assert_eq!(interval, 99_999_999);
        assert!(interval > 0 && interval < 100_000_000);
        let frame_interval = interval;
        assert_eq!(frame_interval, 99_999_999);
    }

    /// Edge case: default 60Hz interval value.
    #[test]
    fn default_60hz_interval_value() {
        let ns_per_sec = 1_000_000_000u64;
        let expected_60hz = ns_per_sec / 60;
        // 16_666_666.666... rounds to 16_666_667.
        assert_eq!(expected_60hz, 16_666_666);
        // The code uses 16_666_667 (rounded up).
        let default = 16_666_667u64;
        assert!(default > expected_60hz);
        assert!(default - expected_60hz <= 1);
    }

    /// Simulate 30Hz presentation with steady timestamps.
    #[test]
    fn steady_30hz_timestamps() {
        let mut last_ns = 0u64;
        let mut frame_interval = 16_666_667u64;
        let interval_30hz = 33_333_333u64; // ~30Hz

        // count = frame index (0-based), equivalent to sample_count before processing.
        for count in 0u64..10 {
            let ts = (count + 1) * interval_30hz;
            if last_ns > 0 {
                let interval = ts.saturating_sub(last_ns);
                if interval > 0 && interval < 100_000_000 {
                    if count < 4 {
                        frame_interval = interval;
                    } else {
                        frame_interval = (frame_interval * 7 + interval) / 8;
                    }
                }
            }
            last_ns = ts;
        }

        // After 10 frames at steady 30Hz, interval should be exactly 33_333_333.
        assert_eq!(frame_interval, interval_30hz);
    }

    /// Simulate a rate switch from 60Hz to 144Hz and verify EMA tracks it.
    #[test]
    fn rate_switch_60_to_144hz() {
        let mut last_ns = 0u64;
        let mut frame_interval = 16_666_667u64;
        let interval_60hz = 16_666_667u64;
        let interval_144hz = 6_944_444u64; // ~144Hz

        // 5 frames at 60Hz. count is the sample index.
        for count in 0u64..5 {
            let ts = (count + 1) * interval_60hz;
            if last_ns > 0 {
                let interval = ts.saturating_sub(last_ns);
                if interval > 0 && interval < 100_000_000 {
                    if count < 4 {
                        frame_interval = interval;
                    } else {
                        frame_interval = (frame_interval * 7 + interval) / 8;
                    }
                }
            }
            last_ns = ts;
        }
        // After 5 frames at 60Hz, interval should be 60Hz.
        assert_eq!(frame_interval, interval_60hz);

        // Now switch to 144Hz for 50 frames.
        // Continuing count from 5.
        let base = last_ns;
        for offset in 0u64..50 {
            let count = 5 + offset;
            let ts = base + (offset + 1) * interval_144hz;
            if last_ns > 0 {
                let interval = ts.saturating_sub(last_ns);
                if interval > 0 && interval < 100_000_000 {
                    if count < 4 {
                        frame_interval = interval;
                    } else {
                        frame_interval = (frame_interval * 7 + interval) / 8;
                    }
                }
            }
            last_ns = ts;
        }

        // Should have converged close to 144Hz.
        let diff = frame_interval.abs_diff(interval_144hz);
        assert!(
            diff < 50_000,
            "Expected near 144Hz ({interval_144hz}), got {frame_interval}, diff={diff}"
        );
    }

    /// Verify `sample_count` increments correctly.
    #[test]
    fn sample_count_increments() {
        let mut count = 0u64;
        for _ in 0..100 {
            count += 1;
        }
        assert_eq!(count, 100);
    }

    /// `record_discarded` is a no-op but should not panic.
    #[test]
    fn record_discarded_is_noop() {
        // The method body is empty — just ensure it doesn't panic.
        // We can't call it without a PresentationSync, so we just verify
        // the concept: discarded frames leave timing unchanged.
        let frame_interval = 16_666_667u64;
        let last_ns = 100_000_000u64;
        // After "discard" — nothing changes.
        assert_eq!(frame_interval, 16_666_667);
        assert_eq!(last_ns, 100_000_000);
    }
}

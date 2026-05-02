/// Application-level backpressure for outgoing frame chunks.
///
/// Tracks pending bytes per stream. When adding a chunk would exceed the
/// high-watermark it is dropped; anchor frames are never dropped.
pub struct FlowController {
    high_watermark: usize,
    pending: usize,
}

impl FlowController {
    /// Create a new flow controller with the given high-watermark in bytes.
    #[must_use]
    pub fn new(high_watermark: usize) -> Self {
        Self {
            high_watermark,
            pending: 0,
        }
    }

    /// Record that `bytes` are about to be enqueued for sending.
    /// Returns `true` if they should be sent, `false` if they should be dropped.
    /// Anchor frames (`is_anchor = true`) are never dropped.
    pub fn should_send(&mut self, bytes: usize, is_anchor: bool) -> bool {
        if is_anchor {
            self.pending += bytes;
            return true;
        }
        // Reject if adding `bytes` would exceed the high-watermark.
        if self.pending.saturating_add(bytes) > self.high_watermark {
            return false;
        }
        self.pending += bytes;
        true
    }

    /// Acknowledge that `bytes` have been sent and removed from the queue.
    ///
    /// Uses saturating arithmetic — it is safe to call with more bytes than
    /// are currently pending.
    pub fn on_sent(&mut self, bytes: usize) {
        self.pending = self.pending.saturating_sub(bytes);
    }

    /// Current number of pending bytes.
    #[must_use]
    pub fn pending_bytes(&self) -> usize {
        self.pending
    }

    /// Whether the pending bytes meet or exceed the high-watermark.
    #[must_use]
    pub fn is_congested(&self) -> bool {
        self.pending >= self.high_watermark
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allows_under_watermark() {
        let mut fc = FlowController::new(1024);
        assert!(fc.should_send(512, false)); // pending = 512
        assert!(fc.should_send(511, false)); // pending = 1023
        // 1023 + 2 = 1025 > 1024 → reject.
        assert!(!fc.should_send(2, false));
        // 1023 + 1 = 1024 ≤ 1024 → allow.
        assert!(fc.should_send(1, false));
    }

    #[test]
    fn anchor_always_passes() {
        let mut fc = FlowController::new(100);
        // Fill to capacity.
        assert!(fc.should_send(100, false)); // pending = 100
        assert!(fc.is_congested());
        // Non-anchor is rejected over the watermark.
        assert!(!fc.should_send(1, false));
        // Anchor always passes regardless.
        assert!(fc.should_send(1, true));
    }

    #[test]
    fn on_sent_drains_pending() {
        let mut fc = FlowController::new(100);
        fc.should_send(100, false);
        assert!(fc.is_congested());
        fc.on_sent(100);
        assert!(!fc.is_congested());
        assert!(fc.should_send(50, false));
    }

    #[test]
    fn saturating_sub_no_underflow() {
        let mut fc = FlowController::new(100);
        fc.on_sent(9999);
        assert_eq!(fc.pending_bytes(), 0);
    }

    #[test]
    fn is_congested_systematic() {
        let mut fc = FlowController::new(100);
        assert!(!fc.is_congested());
        fc.should_send(100, false);
        assert!(fc.is_congested());
        fc.on_sent(1);
        assert!(!fc.is_congested());
        fc.should_send(1, false);
        assert!(fc.is_congested());
    }

    #[test]
    fn rapid_cycle_1000() {
        let mut fc = FlowController::new(1_000_000);
        for _ in 0..1000 {
            assert!(fc.should_send(10, false));
            fc.on_sent(10);
        }
        assert_eq!(fc.pending_bytes(), 0);
    }

    #[test]
    fn zero_watermark() {
        let mut fc = FlowController::new(0);
        // 0 + 0 = 0 ≤ 0 → allow.
        assert!(fc.should_send(0, false));
        // 0 + 1 = 1 > 0 → reject.
        assert!(!fc.should_send(1, false));
    }

    #[test]
    fn anchor_accumulates_beyond_watermark() {
        let mut fc = FlowController::new(100);
        fc.should_send(100, false);
        assert!(fc.should_send(200, true)); // anchor always passes
        assert_eq!(fc.pending_bytes(), 300);
    }
}

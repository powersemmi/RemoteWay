use std::time::{Duration, Instant};

/// Sliding-window bandwidth meter.
///
/// Measures bytes per second over a configurable time window. Safe for
/// single-thread use on a pipeline stage.
#[derive(Debug)]
pub struct BandwidthMeter {
    /// Ring buffer of (timestamp, bytes) samples.
    samples: Vec<(Instant, u64)>,
    cursor: usize,
    label: String,
    /// Window duration over which to calculate bandwidth.
    window: Duration,
    /// Total bytes in the current window.
    window_bytes: u64,
}

/// Bandwidth snapshot.
#[derive(Debug, Clone)]
pub struct BandwidthStats {
    pub label: String,
    /// Bytes per second over the measurement window.
    pub bytes_per_sec: f64,
    /// Megabits per second (bytes_per_sec * 8 / 1_000_000).
    pub mbps: f64,
    /// Total bytes recorded.
    pub total_bytes: u64,
}

const DEFAULT_CAPACITY: usize = 256;

impl BandwidthMeter {
    /// Create a new meter with a 1-second measurement window.
    pub fn new(label: impl Into<String>) -> Self {
        Self::with_window(label, Duration::from_secs(1))
    }

    /// Create a new meter with a custom measurement window.
    pub fn with_window(label: impl Into<String>, window: Duration) -> Self {
        Self {
            samples: Vec::with_capacity(DEFAULT_CAPACITY),
            cursor: 0,
            label: label.into(),
            window,
            window_bytes: 0,
        }
    }

    /// Record that `bytes` were transferred at the current instant.
    pub fn record(&mut self, bytes: u64) {
        self.record_at(Instant::now(), bytes);
    }

    /// Record that `bytes` were transferred at the given instant.
    pub fn record_at(&mut self, now: Instant, bytes: u64) {
        if self.samples.len() < self.samples.capacity() {
            self.samples.push((now, bytes));
        } else {
            self.samples[self.cursor] = (now, bytes);
        }
        self.cursor = (self.cursor + 1) % self.samples.capacity().max(1);
        self.window_bytes += bytes;
    }

    /// Get current bandwidth statistics.
    pub fn stats(&self) -> BandwidthStats {
        let now = Instant::now();
        let cutoff = now.checked_sub(self.window).unwrap_or(now);

        let window_bytes: u64 = self
            .samples
            .iter()
            .filter(|(ts, _)| *ts >= cutoff)
            .map(|(_, b)| *b)
            .sum();

        let elapsed = self.window.as_secs_f64();
        let bytes_per_sec = if elapsed > 0.0 {
            window_bytes as f64 / elapsed
        } else {
            0.0
        };

        BandwidthStats {
            label: self.label.clone(),
            bytes_per_sec,
            mbps: bytes_per_sec * 8.0 / 1_000_000.0,
            total_bytes: self.window_bytes,
        }
    }

    /// Reset all statistics.
    pub fn reset(&mut self) {
        self.samples.clear();
        self.cursor = 0;
        self.window_bytes = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_meter() {
        let m = BandwidthMeter::new("test");
        let s = m.stats();
        assert_eq!(s.total_bytes, 0);
        assert_eq!(s.bytes_per_sec, 0.0);
    }

    #[test]
    fn single_record() {
        let mut m = BandwidthMeter::new("test");
        m.record(1024);
        let s = m.stats();
        assert_eq!(s.total_bytes, 1024);
        assert!(s.bytes_per_sec > 0.0);
    }

    #[test]
    fn multiple_records() {
        let mut m = BandwidthMeter::new("test");
        let now = Instant::now();
        m.record_at(now, 1000);
        m.record_at(now, 2000);
        m.record_at(now, 3000);
        let s = m.stats();
        assert_eq!(s.total_bytes, 6000);
    }

    #[test]
    fn mbps_calculation() {
        let m = BandwidthMeter::new("test");
        let s = m.stats();
        assert_eq!(s.mbps, s.bytes_per_sec * 8.0 / 1_000_000.0);
    }

    #[test]
    fn reset_clears() {
        let mut m = BandwidthMeter::new("test");
        m.record(1024);
        m.reset();
        assert_eq!(m.stats().total_bytes, 0);
    }
}

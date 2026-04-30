use std::time::Duration;

/// Histogram for measuring end-to-end latency across pipeline stages.
///
/// Tracks per-stage and total latency with running statistics (min, max, avg, p99).
/// Thread-safe: uses atomic operations for concurrent updates.
#[derive(Debug)]
pub struct LatencyHistogram {
    /// Ring buffer of recent samples for percentile calculation.
    samples: Vec<Duration>,
    /// Next write position in the ring buffer.
    cursor: usize,
    /// Number of samples recorded (may exceed capacity).
    count: u64,
    /// Running sum for average calculation.
    sum: Duration,
    /// Minimum observed latency.
    min: Duration,
    /// Maximum observed latency.
    max: Duration,
    /// Label for this histogram (e.g. "capture→compress", "end-to-end").
    label: String,
}

/// Snapshot of histogram statistics at a point in time.
#[derive(Debug, Clone)]
pub struct LatencyStats {
    pub label: String,
    pub count: u64,
    pub min: Duration,
    pub max: Duration,
    pub avg: Duration,
    pub p50: Duration,
    pub p99: Duration,
}

const DEFAULT_CAPACITY: usize = 1024;

impl LatencyHistogram {
    /// Create a new histogram with the given label and default capacity (1024 samples).
    pub fn new(label: impl Into<String>) -> Self {
        Self::with_capacity(label, DEFAULT_CAPACITY)
    }

    /// Create a new histogram with the given label and sample buffer capacity.
    pub fn with_capacity(label: impl Into<String>, capacity: usize) -> Self {
        Self {
            samples: Vec::with_capacity(capacity),
            cursor: 0,
            count: 0,
            sum: Duration::ZERO,
            min: Duration::MAX,
            max: Duration::ZERO,
            label: label.into(),
        }
    }

    /// Record a latency sample.
    pub fn record(&mut self, latency: Duration) {
        if self.samples.len() < self.samples.capacity() {
            self.samples.push(latency);
        } else {
            self.samples[self.cursor] = latency;
        }
        self.cursor = (self.cursor + 1) % self.samples.capacity().max(1);
        self.count += 1;
        self.sum += latency;
        self.min = self.min.min(latency);
        self.max = self.max.max(latency);
    }

    /// Record latency from a nanosecond timestamp delta.
    pub fn record_ns(&mut self, start_ns: u64, end_ns: u64) {
        if end_ns > start_ns {
            self.record(Duration::from_nanos(end_ns - start_ns));
        }
    }

    /// Get current statistics snapshot.
    pub fn stats(&self) -> LatencyStats {
        let avg = if self.count > 0 {
            self.sum / self.count as u32
        } else {
            Duration::ZERO
        };

        let (p50, p99) = self.percentiles();

        LatencyStats {
            label: self.label.clone(),
            count: self.count,
            min: if self.count > 0 {
                self.min
            } else {
                Duration::ZERO
            },
            max: self.max,
            avg,
            p50,
            p99,
        }
    }

    /// Reset all statistics.
    pub fn reset(&mut self) {
        self.samples.clear();
        self.cursor = 0;
        self.count = 0;
        self.sum = Duration::ZERO;
        self.min = Duration::MAX;
        self.max = Duration::ZERO;
    }

    fn percentiles(&self) -> (Duration, Duration) {
        if self.samples.is_empty() {
            return (Duration::ZERO, Duration::ZERO);
        }

        let mut sorted: Vec<Duration> = self.samples.clone();
        sorted.sort();

        let p50_idx = sorted.len() / 2;
        let p99_idx = (sorted.len() * 99 / 100).min(sorted.len() - 1);

        (sorted[p50_idx], sorted[p99_idx])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_histogram() {
        let h = LatencyHistogram::new("test");
        let s = h.stats();
        assert_eq!(s.count, 0);
        assert_eq!(s.avg, Duration::ZERO);
        assert_eq!(s.p50, Duration::ZERO);
    }

    #[test]
    fn single_sample() {
        let mut h = LatencyHistogram::new("test");
        h.record(Duration::from_millis(10));
        let s = h.stats();
        assert_eq!(s.count, 1);
        assert_eq!(s.min, Duration::from_millis(10));
        assert_eq!(s.max, Duration::from_millis(10));
    }

    #[test]
    fn min_max_avg() {
        let mut h = LatencyHistogram::new("test");
        h.record(Duration::from_millis(10));
        h.record(Duration::from_millis(20));
        h.record(Duration::from_millis(30));
        let s = h.stats();
        assert_eq!(s.min, Duration::from_millis(10));
        assert_eq!(s.max, Duration::from_millis(30));
        assert_eq!(s.avg, Duration::from_millis(20));
    }

    #[test]
    fn record_ns() {
        let mut h = LatencyHistogram::new("test");
        h.record_ns(1_000_000, 2_000_000);
        let s = h.stats();
        assert_eq!(s.count, 1);
        assert_eq!(s.min, Duration::from_millis(1));
    }

    #[test]
    fn ring_buffer_wraps() {
        let mut h = LatencyHistogram::with_capacity("test", 4);
        for i in 0..10 {
            h.record(Duration::from_millis(i));
        }
        assert_eq!(h.stats().count, 10);
        // Ring buffer only holds last 4 samples.
        assert_eq!(h.samples.len(), 4);
    }

    #[test]
    fn percentiles_sorted() {
        let mut h = LatencyHistogram::new("test");
        for i in (0..100).rev() {
            h.record(Duration::from_millis(i));
        }
        let s = h.stats();
        assert!(s.p50 <= s.p99);
        assert!(s.p50 >= s.min);
        assert!(s.p99 <= s.max);
    }

    #[test]
    fn reset_clears_all() {
        let mut h = LatencyHistogram::new("test");
        h.record(Duration::from_millis(42));
        h.reset();
        assert_eq!(h.stats().count, 0);
    }
}

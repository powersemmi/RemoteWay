use std::time::{Duration, Instant};

/// Per-frame compression metrics.
#[derive(Debug, Default, Clone)]
pub struct FrameStats {
    pub original_bytes: usize,
    pub compressed_bytes: usize,
    pub encode_time: Duration,
    pub compress_time: Duration,
}

impl FrameStats {
    pub fn compression_ratio(&self) -> f32 {
        if self.original_bytes == 0 {
            return 1.0;
        }
        self.compressed_bytes as f32 / self.original_bytes as f32
    }

    pub fn total_time(&self) -> Duration {
        self.encode_time + self.compress_time
    }
}

/// Accumulates [`FrameStats`] across multiple frames.
#[derive(Debug, Default)]
pub struct CompressionStats {
    pub frame_count: u64,
    pub total_original: u64,
    pub total_compressed: u64,
    pub total_encode_time: Duration,
    pub total_compress_time: Duration,
}

impl CompressionStats {
    pub fn record(&mut self, frame: &FrameStats) {
        self.frame_count += 1;
        self.total_original += frame.original_bytes as u64;
        self.total_compressed += frame.compressed_bytes as u64;
        self.total_encode_time += frame.encode_time;
        self.total_compress_time += frame.compress_time;
    }

    pub fn avg_ratio(&self) -> f32 {
        if self.total_original == 0 {
            return 1.0;
        }
        self.total_compressed as f32 / self.total_original as f32
    }

    pub fn avg_encode_ms(&self) -> f32 {
        if self.frame_count == 0 {
            return 0.0;
        }
        self.total_encode_time.as_secs_f32() * 1000.0 / self.frame_count as f32
    }

    pub fn avg_compress_ms(&self) -> f32 {
        if self.frame_count == 0 {
            return 0.0;
        }
        self.total_compress_time.as_secs_f32() * 1000.0 / self.frame_count as f32
    }
}

/// Simple wall-clock timer for measuring stage latency.
pub struct StageTimer {
    start: Instant,
}

impl StageTimer {
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compression_ratio_zero_original() {
        let s = FrameStats::default();
        assert_eq!(s.compression_ratio(), 1.0);
    }

    #[test]
    fn compression_ratio_nonzero() {
        let s = FrameStats {
            original_bytes: 1000,
            compressed_bytes: 400,
            ..Default::default()
        };
        assert!((s.compression_ratio() - 0.4).abs() < 1e-6);
    }

    #[test]
    fn total_time_sums_durations() {
        let s = FrameStats {
            encode_time: Duration::from_millis(3),
            compress_time: Duration::from_millis(7),
            ..Default::default()
        };
        assert_eq!(s.total_time(), Duration::from_millis(10));
    }

    #[test]
    fn accumulates_correctly() {
        let mut stats = CompressionStats::default();
        let frame = FrameStats {
            original_bytes: 1000,
            compressed_bytes: 500,
            encode_time: Duration::from_millis(1),
            compress_time: Duration::from_millis(2),
        };
        stats.record(&frame);
        stats.record(&frame);
        assert_eq!(stats.frame_count, 2);
        assert_eq!(stats.total_original, 2000);
        assert_eq!(stats.total_compressed, 1000);
        assert!((stats.avg_ratio() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn avg_ratio_zero_original() {
        let stats = CompressionStats::default();
        assert_eq!(stats.avg_ratio(), 1.0);
    }

    #[test]
    fn avg_timing_zero_frames() {
        let stats = CompressionStats::default();
        assert_eq!(stats.avg_encode_ms(), 0.0);
        assert_eq!(stats.avg_compress_ms(), 0.0);
    }

    #[test]
    fn avg_timing_nonzero() {
        let mut stats = CompressionStats::default();
        let frame = FrameStats {
            original_bytes: 100,
            compressed_bytes: 50,
            encode_time: Duration::from_millis(4),
            compress_time: Duration::from_millis(6),
        };
        stats.record(&frame);
        assert!((stats.avg_encode_ms() - 4.0).abs() < 0.1);
        assert!((stats.avg_compress_ms() - 6.0).abs() < 0.1);
    }

    #[test]
    fn stage_timer_elapsed_nonzero() {
        let timer = StageTimer::start();
        // At least a tiny bit of time passes even with no sleep.
        let elapsed = timer.elapsed();
        // Just verify it's a valid (non-panicking) Duration.
        let _ = elapsed.as_nanos();
    }

    #[test]
    fn frame_stats_default_all_zero() {
        let s = FrameStats::default();
        assert_eq!(s.original_bytes, 0);
        assert_eq!(s.compressed_bytes, 0);
        assert_eq!(s.encode_time, Duration::ZERO);
        assert_eq!(s.compress_time, Duration::ZERO);
    }

    #[test]
    fn frame_stats_clone_equal() {
        let s = FrameStats {
            original_bytes: 100,
            compressed_bytes: 50,
            encode_time: Duration::from_millis(1),
            compress_time: Duration::from_millis(2),
        };
        let cloned = s.clone();
        assert_eq!(cloned.original_bytes, s.original_bytes);
        assert_eq!(cloned.compressed_bytes, s.compressed_bytes);
        assert_eq!(cloned.encode_time, s.encode_time);
        assert_eq!(cloned.compress_time, s.compress_time);
    }

    #[test]
    fn compression_stats_default_frame_count_zero() {
        let stats = CompressionStats::default();
        assert_eq!(stats.frame_count, 0);
        assert_eq!(stats.total_original, 0);
        assert_eq!(stats.total_compressed, 0);
    }
}

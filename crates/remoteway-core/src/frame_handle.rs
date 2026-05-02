/// Lightweight handle passed between pipeline stages instead of copying frame data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct FrameHandle {
    /// Index of the owning buffer within the frame pool.
    pub pool_index: u16,
    /// Length of the frame payload in bytes.
    pub len: u32,
    /// Capture timestamp in nanoseconds (monotonic clock).
    pub timestamp_ns: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_access() {
        let h = FrameHandle {
            pool_index: 7,
            len: 4096,
            timestamp_ns: 123_456_789,
        };
        assert_eq!(h.pool_index, 7);
        assert_eq!(h.len, 4096);
        assert_eq!(h.timestamp_ns, 123_456_789);
    }

    #[test]
    fn copy_semantics() {
        let h = FrameHandle {
            pool_index: 1,
            len: 256,
            timestamp_ns: 100,
        };
        let h2 = h; // Copy
        assert_eq!(h, h2);
        // original is still usable (Copy, not moved)
        assert_eq!(h.pool_index, 1);
    }

    #[test]
    fn clone_equals_original() {
        let h = FrameHandle {
            pool_index: 3,
            len: 512,
            timestamp_ns: 999,
        };
        assert_eq!(h.clone(), h);
    }

    #[test]
    fn debug_format() {
        let h = FrameHandle {
            pool_index: 5,
            len: 1024,
            timestamp_ns: 42,
        };
        let dbg = format!("{:?}", h);
        assert!(dbg.contains("FrameHandle"));
        assert!(dbg.contains("1024"));
        assert!(dbg.contains("42"));
    }

    #[test]
    fn eq_and_ne() {
        let a = FrameHandle {
            pool_index: 0,
            len: 64,
            timestamp_ns: 1,
        };
        let b = FrameHandle {
            pool_index: 0,
            len: 64,
            timestamp_ns: 2,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn zero_values() {
        let h = FrameHandle {
            pool_index: 0,
            len: 0,
            timestamp_ns: 0,
        };
        assert_eq!(h.pool_index, 0);
        assert_eq!(h.len, 0);
        assert_eq!(h.timestamp_ns, 0);
    }

    #[test]
    fn max_values() {
        let h = FrameHandle {
            pool_index: u16::MAX,
            len: u32::MAX,
            timestamp_ns: u64::MAX,
        };
        assert_eq!(h.pool_index, u16::MAX);
        assert_eq!(h.len, u32::MAX);
        assert_eq!(h.timestamp_ns, u64::MAX);
    }
}

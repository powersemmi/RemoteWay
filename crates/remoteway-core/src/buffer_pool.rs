use std::sync::atomic::{AtomicU64, Ordering};

use crate::frame_handle::FrameHandle;

/// Pre-allocated pool of frame buffers aligned to 64 bytes for AVX-512.
///
/// Buffers are managed via a lock-free free-list encoded in a single `AtomicU64`
/// (bitset of available slots, max 64 slots). acquire/release are wait-free on
/// the fast path; only contention on the same slot can loop.
pub struct BufferPool {
    /// Raw backing allocation. Each slot is `frame_size` bytes, aligned to 64.
    storage: *mut u8,
    frame_size: usize,
    capacity: usize,
    /// Bitset: bit N = 1 means slot N is free.
    free_mask: AtomicU64,
}

// SAFETY: BufferPool owns its allocation exclusively; the free_mask ensures
// no two callers hold the same slot simultaneously.
unsafe impl Send for BufferPool {}
// SAFETY: BufferPool uses AtomicU64 for lock-free synchronization.
#[allow(clippy::undocumented_unsafe_blocks)]
unsafe impl Sync for BufferPool {}

impl BufferPool {
    /// Create a pool of `capacity` buffers each `frame_size` bytes.
    /// `capacity` must be ≤ 64. `frame_size` is rounded up to a multiple of 64.
    #[must_use]
    pub fn new(capacity: usize, frame_size: usize) -> Self {
        assert!(capacity > 0 && capacity <= 64, "capacity must be 1..=64");
        let aligned_size = (frame_size + 63) & !63;
        let total = aligned_size * capacity;

        // SAFETY: Layout is non-zero (asserted above). We use alloc_zeroed so
        // buffers start in a defined state. align=64 satisfies AVX-512 loads.
        // SAFETY: total > 0 and 64 is a valid alignment for AVX-512.
        let layout = std::alloc::Layout::from_size_align(total, 64)
            .unwrap_or_else(|_| panic!("BUG: invalid layout: total={total}, align=64"));
        // SAFETY: layout is valid and non-zero (capacity >= 1).
        let storage = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!storage.is_null(), "allocation failed");

        let free_mask = if capacity == 64 {
            u64::MAX
        } else {
            (1u64 << capacity) - 1
        };

        Self {
            storage,
            frame_size: aligned_size,
            capacity,
            free_mask: AtomicU64::new(free_mask),
        }
    }

    /// Acquire a free slot. Returns `None` if the pool is exhausted.
    #[inline]
    pub fn acquire(&self, timestamp_ns: u64) -> Option<FrameHandle> {
        loop {
            let mask = self.free_mask.load(Ordering::Acquire);
            if mask == 0 {
                return None;
            }
            let slot = mask.trailing_zeros() as u16;
            let new_mask = mask & !(1u64 << slot);
            if self
                .free_mask
                .compare_exchange_weak(mask, new_mask, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Some(FrameHandle {
                    pool_index: slot,
                    len: self.frame_size as u32,
                    timestamp_ns,
                });
            }
        }
    }

    /// Release a slot back to the pool.
    ///
    /// # Panics
    /// Panics in debug mode if the slot index is out of range or already free.
    #[inline]
    pub fn release(&self, handle: FrameHandle) {
        let slot = handle.pool_index as u64;
        debug_assert!(
            (slot as usize) < self.capacity,
            "pool_index {} out of range",
            slot
        );
        let bit = 1u64 << slot;
        let prev = self.free_mask.fetch_or(bit, Ordering::Release);
        debug_assert_eq!(prev & bit, 0, "double-release of slot {}", slot);
    }

    /// Get a mutable pointer to the buffer for `handle`.
    ///
    /// # Safety
    /// Caller must hold the handle (i.e., have acquired it and not yet released it).
    #[inline]
    pub unsafe fn slot_ptr(&self, handle: &FrameHandle) -> *mut u8 {
        // SAFETY: slot was acquired and is within capacity bounds.
        unsafe {
            self.storage
                .add(handle.pool_index as usize * self.frame_size)
        }
    }

    /// Get a shared slice for the buffer identified by `handle`.
    ///
    /// # Safety
    /// Caller must hold the handle and not have concurrent mutable access.
    #[inline]
    pub unsafe fn slot(&self, handle: &FrameHandle) -> &[u8] {
        // SAFETY: ptr is valid, aligned, initialized, and handle is exclusively held.
        unsafe {
            std::slice::from_raw_parts(
                self.storage
                    .add(handle.pool_index as usize * self.frame_size),
                handle.len as usize,
            )
        }
    }

    /// Get a mutable slice for the buffer identified by `handle`.
    ///
    /// # Safety
    /// Caller must hold the handle exclusively; no other reference to this slot may exist.
    /// `&self` is used instead of `&mut self` to allow concurrent access to *different* slots.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn slot_mut(&self, handle: &FrameHandle) -> &mut [u8] {
        // SAFETY: ptr is valid, aligned, and the caller guarantees exclusive ownership of this slot.
        unsafe {
            std::slice::from_raw_parts_mut(
                self.storage
                    .add(handle.pool_index as usize * self.frame_size),
                handle.len as usize,
            )
        }
    }

    /// Total number of slots in the pool.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Size of each frame buffer in bytes (aligned up to 64).
    pub fn frame_size(&self) -> usize {
        self.frame_size
    }

    /// Number of currently free slots.
    pub fn free_count(&self) -> u32 {
        self.free_mask.load(Ordering::Relaxed).count_ones()
    }
}

impl Drop for BufferPool {
    fn drop(&mut self) {
        let aligned_size = self.frame_size;
        let total = aligned_size * self.capacity;
        // The layout here mirrors the one used at allocation time in `new`.
        #[allow(clippy::expect_used)]
        let layout = std::alloc::Layout::from_size_align(total, 64)
            .unwrap_or_else(|_| panic!("BUG: invalid layout in drop: total={total}, align=64"));
        // SAFETY: `storage` was allocated with this exact layout in `new`.
        unsafe { std::alloc::dealloc(self.storage, layout) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acquire_and_release_basic() {
        let pool = BufferPool::new(4, 1024);
        assert_eq!(pool.free_count(), 4);

        let h = pool.acquire(0).unwrap();
        assert_eq!(pool.free_count(), 3);
        pool.release(h);
        assert_eq!(pool.free_count(), 4);
    }

    #[test]
    fn exhaust_pool_returns_none() {
        let pool = BufferPool::new(2, 64);
        let h0 = pool.acquire(0).unwrap();
        let h1 = pool.acquire(1).unwrap();
        assert!(pool.acquire(2).is_none());
        pool.release(h0);
        assert!(pool.acquire(3).is_some());
        pool.release(h1);
    }

    #[test]
    fn slot_ptr_is_64_byte_aligned() {
        let pool = BufferPool::new(4, 100);
        let h = pool.acquire(0).unwrap();
        // SAFETY: handle was just acquired and is exclusively held.
        let ptr = unsafe { pool.slot_ptr(&h) };
        assert_eq!(ptr as usize % 64, 0);
        pool.release(h);
    }

    #[test]
    fn frame_size_rounded_to_64() {
        let pool = BufferPool::new(2, 1);
        assert_eq!(pool.frame_size(), 64);
    }

    #[test]
    fn max_capacity_64() {
        let pool = BufferPool::new(64, 64);
        assert_eq!(pool.free_count(), 64);
        let handles: Vec<_> = (0..64).map(|ts| pool.acquire(ts).unwrap()).collect();
        assert!(pool.acquire(64).is_none());
        for h in handles {
            pool.release(h);
        }
        assert_eq!(pool.free_count(), 64);
    }

    #[test]
    fn concurrent_acquire_release() {
        use std::sync::Arc;
        let pool = Arc::new(BufferPool::new(8, 256));
        let mut threads = Vec::new();
        for _ in 0..4 {
            let p = Arc::clone(&pool);
            threads.push(std::thread::spawn(move || {
                for ts in 0..100u64 {
                    if let Some(h) = p.acquire(ts) {
                        p.release(h);
                    }
                }
            }));
        }
        for t in threads {
            t.join().unwrap();
        }
        assert_eq!(pool.free_count(), 8);
    }

    #[test]
    fn capacity_one_acquire_release() {
        let pool = BufferPool::new(1, 64);
        assert_eq!(pool.free_count(), 1);
        let h = pool.acquire(0).unwrap();
        assert!(pool.acquire(1).is_none());
        pool.release(h);
        let h2 = pool.acquire(2).unwrap();
        assert_eq!(h2.pool_index, 0);
        pool.release(h2);
    }

    #[test]
    fn slot_data_write_read_round_trip() {
        let pool = BufferPool::new(2, 64);
        let h = pool.acquire(0).unwrap();
        let pattern = [0xAB_u8; 64];
        // SAFETY: handle is exclusively held by this test.
        unsafe {
            pool.slot_mut(&h).copy_from_slice(&pattern);
            assert_eq!(pool.slot(&h), &pattern);
        }
        pool.release(h);
    }

    #[test]
    fn two_slots_independent_data() {
        let pool = BufferPool::new(4, 64);
        let h0 = pool.acquire(0).unwrap();
        let h1 = pool.acquire(1).unwrap();
        // SAFETY: both handles are exclusively held by this test.
        unsafe {
            pool.slot_mut(&h0).fill(0x11);
            pool.slot_mut(&h1).fill(0x22);
            assert!(pool.slot(&h0).iter().all(|&b| b == 0x11));
            assert!(pool.slot(&h1).iter().all(|&b| b == 0x22));
        }
        pool.release(h0);
        pool.release(h1);
    }

    #[test]
    fn free_count_tracks_correctly() {
        let pool = BufferPool::new(4, 64);
        assert_eq!(pool.free_count(), 4);
        let h0 = pool.acquire(0).unwrap();
        assert_eq!(pool.free_count(), 3);
        let h1 = pool.acquire(1).unwrap();
        assert_eq!(pool.free_count(), 2);
        let h2 = pool.acquire(2).unwrap();
        assert_eq!(pool.free_count(), 1);
        let h3 = pool.acquire(3).unwrap();
        assert_eq!(pool.free_count(), 0);
        pool.release(h3);
        assert_eq!(pool.free_count(), 1);
        pool.release(h2);
        assert_eq!(pool.free_count(), 2);
        pool.release(h1);
        assert_eq!(pool.free_count(), 3);
        pool.release(h0);
        assert_eq!(pool.free_count(), 4);
    }

    #[test]
    fn frame_size_alignment_various_inputs() {
        for &input in &[1, 63, 64, 65, 128, 129, 255, 256] {
            let pool = BufferPool::new(1, input);
            assert_eq!(
                pool.frame_size() % 64,
                0,
                "frame_size({}) not aligned",
                input
            );
            assert!(pool.frame_size() >= input);
        }
    }

    #[test]
    fn capacity_method_returns_configured() {
        let pool = BufferPool::new(17, 128);
        assert_eq!(pool.capacity(), 17);
    }
}

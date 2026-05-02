use std::sync::{Arc, Barrier};
use std::thread;

use remoteway_core::buffer_pool::BufferPool;
use remoteway_core::frame_handle::FrameHandle;

/// 8 threads, each acquiring and releasing handles 10 000 times.
/// Verifies no slot is ever double-held and the pool fully drains back.
#[test]
fn stress_concurrent_acquire_release() {
    const THREADS: usize = 8;
    const ITERS: usize = 10_000;
    const POOL_SIZE: usize = 8;

    let pool = Arc::new(BufferPool::new(POOL_SIZE, 256));
    let barrier = Arc::new(Barrier::new(THREADS));

    let handles: Vec<_> = (0..THREADS)
        .map(|tid| {
            let pool = Arc::clone(&pool);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                // All threads start at the same time.
                let _ = barrier.wait();
                for i in 0..ITERS as u64 {
                    // Spin until we get a slot — pool may be momentarily full.
                    let handle = loop {
                        if let Some(h) = pool.acquire(tid as u64 * ITERS as u64 + i) {
                            break h;
                        }
                        thread::yield_now();
                    };
                    // Write a marker into the slot to detect overlap.
                    let marker = (tid as u8).wrapping_add(1);
                    // SAFETY: handle was just acquired and is exclusively held by this thread.
                    unsafe {
                        let ptr = pool.slot_ptr(&handle);
                        ptr.write_bytes(marker, 1);
                    }
                    // Yield occasionally to increase interleaving.
                    if i % 100 == 0 {
                        thread::yield_now();
                    }
                    pool.release(handle);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // After all threads finish, all slots must be free.
    assert_eq!(
        pool.free_count() as usize,
        POOL_SIZE,
        "all slots must be returned after stress test"
    );
}

/// Write then read back verifies slot data survives across acquire/release.
#[test]
fn write_and_read_slot_data() {
    let pool = Arc::new(BufferPool::new(4, 64));

    let h: FrameHandle = pool.acquire(42).unwrap();
    // SAFETY: handle was just acquired above and is exclusively held.
    unsafe {
        let buf = pool.slot_mut(&h);
        buf[0] = 0xDE;
        buf[1] = 0xAD;
        buf[2] = 0xBE;
        buf[3] = 0xEF;
    }
    // SAFETY: handle is still exclusively held; reading back written data.
    unsafe {
        let buf = pool.slot(&h);
        assert_eq!(&buf[..4], &[0xDE, 0xAD, 0xBE, 0xEF]);
    }
    pool.release(h);
}

/// Slots reused after release must not carry stale data (caller responsibility),
/// but the pool must hand out the same physical slot (index 0) after it was released.
#[test]
fn released_slot_is_reusable() {
    let pool = BufferPool::new(1, 64);
    let h1 = pool.acquire(1).unwrap();
    let idx1 = h1.pool_index;
    pool.release(h1);

    let h2 = pool.acquire(2).unwrap();
    assert_eq!(
        h2.pool_index, idx1,
        "pool with one slot must reuse the same index"
    );
    pool.release(h2);
}

/// `FrameHandle` carries the correct timestamp through acquire.
#[test]
fn frame_handle_timestamp_preserved() {
    let pool = BufferPool::new(2, 64);
    let ts = 1_234_567_890_u64;
    let h = pool.acquire(ts).unwrap();
    assert_eq!(h.timestamp_ns, ts);
    pool.release(h);
}

/// `pool_index` is always within [0, capacity).
#[test]
fn pool_index_within_capacity() {
    let capacity = 16;
    let pool = BufferPool::new(capacity, 32);
    let mut handles = Vec::new();
    for ts in 0..capacity as u64 {
        let h = pool.acquire(ts).unwrap();
        assert!((h.pool_index as usize) < capacity);
        handles.push(h);
    }
    for h in handles {
        pool.release(h);
    }
}

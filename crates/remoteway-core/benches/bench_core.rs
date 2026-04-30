use std::sync::Arc;
use std::thread;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use remoteway_core::buffer_pool::BufferPool;

/// Single-threaded acquire/release throughput for various pool sizes.
fn bench_buffer_pool_single_thread(c: &mut Criterion) {
    let mut group = c.benchmark_group("BufferPool/single_thread");

    for pool_size in [4usize, 8, 16, 32, 64] {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("acquire_release", pool_size),
            &pool_size,
            |b, &size| {
                let pool = BufferPool::new(size, 256);
                let mut ts: u64 = 0;
                b.iter(|| {
                    let h = pool.acquire(ts).unwrap();
                    pool.release(h);
                    ts += 1;
                });
            },
        );
    }

    group.finish();
}

/// Multi-threaded acquire/release: N threads competing for a pool of N slots.
fn bench_buffer_pool_multi_thread(c: &mut Criterion) {
    let mut group = c.benchmark_group("BufferPool/multi_thread");

    for threads in [2usize, 4, 8] {
        group.throughput(Throughput::Elements(threads as u64 * 1000));
        group.bench_with_input(BenchmarkId::new("threads", threads), &threads, |b, &n| {
            b.iter(|| {
                let pool = Arc::new(BufferPool::new(n, 256));
                let handles: Vec<_> = (0..n)
                    .map(|tid| {
                        let pool = Arc::clone(&pool);
                        thread::spawn(move || {
                            for i in 0..1000u64 {
                                let h = loop {
                                    if let Some(h) = pool.acquire(tid as u64 * 1000 + i) {
                                        break h;
                                    }
                                    thread::yield_now();
                                };
                                pool.release(h);
                            }
                        })
                    })
                    .collect();
                for h in handles {
                    h.join().unwrap();
                }
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_buffer_pool_single_thread,
    bench_buffer_pool_multi_thread,
);
criterion_main!(benches);

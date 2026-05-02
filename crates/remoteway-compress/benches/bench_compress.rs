use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use remoteway_compress::delta::{DamageRect, delta_encode_scalar, delta_encode_simd};
use remoteway_compress::lz4::compress_region;
use remoteway_compress::pipeline::compress_frame;

// 4K resolution: 3840×2160, RGBA.
const W4K: usize = 3840;
const H4K: usize = 2160;
const STRIDE4K: usize = W4K * 4;

fn make_frame_4k(seed: u8) -> Vec<u8> {
    (0..W4K * H4K * 4)
        .map(|i| i.wrapping_mul(seed as usize).wrapping_add(i >> 2) as u8)
        .collect()
}

/// Damage rects covering ~5% of a 4K frame (a handful of 200×100 rects).
fn damage_rects_5pct() -> Vec<DamageRect> {
    // 5% of 3840×2160 ≈ 415 800 pixels. Use 20 rects of 200×100 = 400 000 pixels.
    (0..20)
        .map(|i| {
            let x = (i * 192) % (W4K as u32 - 200);
            let y = (i * 100) % (H4K as u32 - 100);
            DamageRect::new(x, y, 200, 100)
        })
        .collect()
}

/// `delta_encode` (SIMD) on a 4K frame with ~5% damage.
fn bench_delta_encode_4k_5pct(c: &mut Criterion) {
    let current = make_frame_4k(7);
    let previous = make_frame_4k(3);
    let regions = damage_rects_5pct();
    let damaged_bytes: u64 = regions
        .iter()
        .map(|r| r.width as u64 * r.height as u64 * 4)
        .sum();

    let mut group = c.benchmark_group("delta_encode");
    group.throughput(Throughput::Bytes(damaged_bytes));

    group.bench_function("4k_5pct_simd", |b| {
        b.iter(|| {
            let mut out = Vec::new();
            delta_encode_simd(&current, &previous, STRIDE4K, &regions, &mut out)
        });
    });

    group.bench_function("4k_5pct_scalar", |b| {
        b.iter(|| {
            let mut out = Vec::new();
            delta_encode_scalar(&current, &previous, STRIDE4K, &regions, &mut out)
        });
    });

    group.finish();
}

/// LZ4 compress a full 4K frame worth of data (simulate full-damage).
fn bench_compress_4k_full(c: &mut Criterion) {
    // Pre-compute the delta (all-XOR vs zero previous = the frame itself).
    let frame = make_frame_4k(5);
    let previous = vec![0u8; W4K * H4K * 4];
    let regions = vec![DamageRect::new(0, 0, W4K as u32, H4K as u32)];
    let mut delta = Vec::new();
    delta_encode_scalar(&frame, &previous, STRIDE4K, &regions, &mut delta);

    let total_bytes = delta.len() as u64;

    let mut group = c.benchmark_group("lz4_compress");
    group.throughput(Throughput::Bytes(total_bytes));

    group.bench_function("4k_full_frame", |b| b.iter(|| compress_region(&delta)));

    group.finish();
}

/// `compress_frame` pipeline (delta + LZ4) for different damage levels.
fn bench_pipeline(c: &mut Criterion) {
    let current = make_frame_4k(11);
    let previous = make_frame_4k(13);

    let mut group = c.benchmark_group("compress_pipeline");

    for (label, regions) in [
        ("4k_5pct", damage_rects_5pct()),
        (
            "4k_full",
            vec![DamageRect::new(0, 0, W4K as u32, H4K as u32)],
        ),
    ] {
        let damaged_bytes: u64 = regions
            .iter()
            .map(|r| r.width as u64 * r.height as u64 * 4)
            .sum();
        group.throughput(Throughput::Bytes(damaged_bytes));
        group.bench_with_input(
            BenchmarkId::new("delta_lz4", label),
            &regions,
            |b, regions| b.iter(|| compress_frame(&current, &previous, STRIDE4K, regions)),
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_delta_encode_4k_5pct,
    bench_compress_4k_full,
    bench_pipeline,
);
criterion_main!(benches);

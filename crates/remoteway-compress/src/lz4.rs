use lz4_flex::{compress_prepend_size, decompress_size_prepended};
use rayon::prelude::*;
use thiserror::Error;

use crate::delta::DamageRect;

#[derive(Debug, Error)]
pub enum CompressError {
    #[error("decompression failed: {0}")]
    Decompress(#[from] lz4_flex::block::DecompressError),
}

/// Compress a single region's delta bytes. Returns the compressed bytes.
pub fn compress_region(data: &[u8]) -> Vec<u8> {
    compress_prepend_size(data)
}

/// Decompress a single region. Returns the original delta bytes.
pub fn decompress_region(data: &[u8]) -> Result<Vec<u8>, CompressError> {
    Ok(decompress_size_prepended(data)?)
}

/// Delta-encode and compress each damage region in parallel via rayon.
///
/// Returns a `Vec` of compressed blobs, one per `regions` entry, in the same order.
/// Each blob is the LZ4-compressed XOR delta for that region.
pub fn compress_regions_parallel(
    current: &[u8],
    previous: &[u8],
    regions: &[DamageRect],
    stride: usize,
) -> Vec<Vec<u8>> {
    // Extract per-region deltas (single-threaded; regions are usually few).
    let deltas: Vec<Vec<u8>> = regions
        .iter()
        .map(|rect| delta_encode_region(current, previous, rect, stride))
        .collect();

    // Compress in parallel.
    deltas.par_iter().map(|d| compress_region(d)).collect()
}

/// XOR-encode a single damage region from `current` vs `previous`.
fn delta_encode_region(
    current: &[u8],
    previous: &[u8],
    rect: &DamageRect,
    stride: usize,
) -> Vec<u8> {
    let row_bytes = rect.width as usize * 4;
    let mut out = Vec::with_capacity(row_bytes * rect.height as usize);
    for row in 0..rect.height as usize {
        let y = rect.y as usize + row;
        let x_start = rect.x as usize * 4;
        let row_start = y * stride + x_start;
        let row_end = row_start + row_bytes;
        out.extend(
            current[row_start..row_end]
                .iter()
                .zip(previous[row_start..row_end].iter())
                .map(|(c, p)| c ^ p),
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::delta::{DamageRect, delta_encode_scalar};

    #[test]
    fn round_trip_single_region() {
        let data: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let compressed = compress_region(&data);
        let decompressed = decompress_region(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn empty_data_compresses_and_decompresses() {
        let compressed = compress_region(&[]);
        let decompressed = decompress_region(&compressed).unwrap();
        assert!(decompressed.is_empty());
    }

    #[test]
    fn parallel_compress_matches_sequential() {
        let w = 16usize;
        let h = 16usize;
        let stride = w * 4;
        let current: Vec<u8> = (0..w * h * 4).map(|i| (i * 7) as u8).collect();
        let previous: Vec<u8> = (0..w * h * 4).map(|i| (i * 3) as u8).collect();
        let regions = vec![
            DamageRect::new(0, 0, 4, 4),
            DamageRect::new(4, 4, 4, 4),
            DamageRect::new(8, 8, 8, 8),
        ];

        let mut delta = Vec::new();
        delta_encode_scalar(&current, &previous, stride, &regions, &mut delta);

        // Sequential
        let seq: Vec<Vec<u8>> = regions
            .iter()
            .map(|r| {
                let mut extracted = Vec::new();
                delta_encode_scalar(&current, &previous, stride, &[*r], &mut extracted);
                compress_region(&extracted)
            })
            .collect();

        // Parallel (delta-encode + compress per region in parallel)
        let par = compress_regions_parallel(&current, &previous, &regions, stride);

        // Both should decompress to the same data.
        for (s, p) in seq.iter().zip(par.iter()) {
            let s_dec = decompress_region(s).unwrap();
            let p_dec = decompress_region(p).unwrap();
            assert_eq!(s_dec, p_dec);
        }
    }

    #[test]
    fn compressed_is_smaller_for_zeroes() {
        let data = vec![0u8; 4096];
        let compressed = compress_region(&data);
        assert!(compressed.len() < data.len());
    }

    #[test]
    fn many_regions_100_parallel() {
        let w = 64usize;
        let h = 64usize;
        let stride = w * 4;
        let current: Vec<u8> = (0..w * h * 4).map(|i| (i * 7) as u8).collect();
        let previous: Vec<u8> = (0..w * h * 4).map(|i| (i * 3) as u8).collect();
        // 100 small 2×2 regions within bounds.
        let regions: Vec<DamageRect> = (0..100)
            .map(|i| DamageRect::new((i % 32) as u32, (i / 32) as u32, 2, 2))
            .collect();
        let blobs = compress_regions_parallel(&current, &previous, &regions, stride);
        assert_eq!(blobs.len(), 100);
        for blob in &blobs {
            let decompressed = decompress_region(blob).unwrap();
            assert_eq!(decompressed.len(), 2 * 2 * 4);
        }
    }

    #[test]
    fn single_region_parallel() {
        let w = 4usize;
        let h = 4usize;
        let stride = w * 4;
        let current: Vec<u8> = (0..w * h * 4).map(|i| i as u8).collect();
        let previous = vec![0u8; w * h * 4];
        let regions = vec![DamageRect::new(0, 0, w as u32, h as u32)];
        let blobs = compress_regions_parallel(&current, &previous, &regions, stride);
        assert_eq!(blobs.len(), 1);
    }

    #[test]
    fn zero_regions_parallel() {
        let current = vec![0u8; 64];
        let previous = vec![0u8; 64];
        let blobs = compress_regions_parallel(&current, &previous, &[], 16);
        assert!(blobs.is_empty());
    }

    #[test]
    fn large_data_round_trip() {
        let data: Vec<u8> = (0..1_048_576).map(|i| (i * 13) as u8).collect();
        let compressed = compress_region(&data);
        let decompressed = decompress_region(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }
}

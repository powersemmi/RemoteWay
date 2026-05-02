//! Unified encode/decode pipeline orchestrating delta + compression stages.

use crate::delta::{DamageRect, delta_decode, delta_encode_simd};
use crate::lz4::{CompressError, compress_region, decompress_region};
use crate::stats::{FrameStats, StageTimer};

/// Output of the compression pipeline for one frame.
#[must_use]
pub struct CompressedFrame {
    /// Concatenated compressed region blobs.
    pub data: Vec<u8>,
    /// Byte offset of each region's blob within `data`.
    pub region_offsets: Vec<usize>,
    /// Sizes of each compressed region blob.
    pub region_sizes: Vec<usize>,
    /// Timing and size metrics for this frame.
    pub stats: FrameStats,
}

/// Encode and compress damaged regions of a frame.
///
/// `current` and `previous` must be the same length (width × height × 4 bytes for RGBA).
/// `stride` is the number of bytes per row (`width * 4`).
pub fn compress_frame(
    current: &[u8],
    previous: &[u8],
    stride: usize,
    regions: &[DamageRect],
) -> CompressedFrame {
    #[cfg(feature = "tracy")]
    let _zone = tracy_client::span!("compress_frame");

    let encode_timer = StageTimer::start();
    let mut delta = Vec::with_capacity(current.len() / 4);
    let original_bytes = delta_encode_simd(current, previous, stride, regions, &mut delta);
    let encode_time = encode_timer.elapsed();

    let compress_timer = StageTimer::start();
    let mut data = Vec::new();
    let mut region_offsets = Vec::with_capacity(regions.len());
    let mut region_sizes = Vec::with_capacity(regions.len());

    // Compress each region independently (could be parallelised via rayon later).
    let mut delta_offset = 0;
    for rect in regions {
        let row_bytes = rect.width as usize * 4;
        let region_len = row_bytes * rect.height as usize;
        let region_delta = &delta[delta_offset..delta_offset + region_len];
        let compressed = compress_region(region_delta);
        region_offsets.push(data.len());
        region_sizes.push(compressed.len());
        data.extend_from_slice(&compressed);
        delta_offset += region_len;
    }
    let compress_time = compress_timer.elapsed();

    let compressed_bytes = data.len();
    CompressedFrame {
        data,
        region_offsets,
        region_sizes,
        stats: FrameStats {
            original_bytes,
            compressed_bytes,
            encode_time,
            compress_time,
        },
    }
}

/// Decompress and reconstruct a frame from a [`CompressedFrame`] and the previous frame.
pub fn decompress_frame(
    compressed: &CompressedFrame,
    previous: &[u8],
    stride: usize,
    regions: &[DamageRect],
    output: &mut Vec<u8>,
) -> Result<(), CompressError> {
    output.resize(previous.len(), 0);

    let mut delta = Vec::new();
    for (i, _rect) in regions.iter().enumerate() {
        let start = compressed.region_offsets[i];
        let end = start + compressed.region_sizes[i];
        let blob = &compressed.data[start..end];
        let region_delta = decompress_region(blob)?;
        delta.extend_from_slice(&region_delta);
    }

    delta_decode(&delta, previous, stride, regions, output);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(w: usize, h: usize) -> Vec<u8> {
        (0..w * h * 4).map(|i| (i * 7 + 3) as u8).collect()
    }

    #[test]
    fn round_trip_full_frame() {
        let w = 16usize;
        let h = 16usize;
        let stride = w * 4;
        let current = make_frame(w, h);
        let previous = vec![0u8; w * h * 4];
        let regions = vec![DamageRect::new(0, 0, w as u32, h as u32)];

        let compressed = compress_frame(&current, &previous, stride, &regions);
        let mut reconstructed = Vec::new();
        decompress_frame(&compressed, &previous, stride, &regions, &mut reconstructed).unwrap();
        assert_eq!(reconstructed, current);
    }

    #[test]
    fn empty_regions_produces_empty_output() {
        let w = 8usize;
        let h = 8usize;
        let stride = w * 4;
        let current = make_frame(w, h);
        let previous = make_frame(w, h);
        let compressed = compress_frame(&current, &previous, stride, &[]);
        assert!(compressed.data.is_empty());
        assert_eq!(compressed.stats.original_bytes, 0);
    }

    #[test]
    fn stats_populated() {
        let w = 8usize;
        let h = 8usize;
        let stride = w * 4;
        let current = make_frame(w, h);
        let previous = vec![0u8; w * h * 4];
        let regions = vec![DamageRect::new(0, 0, w as u32, h as u32)];
        let f = compress_frame(&current, &previous, stride, &regions);
        assert!(f.stats.original_bytes > 0);
        assert!(f.stats.compressed_bytes > 0);
    }

    #[test]
    fn region_offsets_sizes_consistent() {
        let w = 16usize;
        let h = 16usize;
        let stride = w * 4;
        let current = make_frame(w, h);
        let previous = vec![0u8; w * h * 4];
        let regions = vec![
            DamageRect::new(0, 0, 4, 4),
            DamageRect::new(4, 4, 4, 4),
            DamageRect::new(8, 8, 4, 4),
        ];
        let compressed = compress_frame(&current, &previous, stride, &regions);
        assert_eq!(compressed.region_offsets.len(), 3);
        assert_eq!(compressed.region_sizes.len(), 3);
        for i in 0..3 {
            assert!(
                compressed.region_offsets[i] + compressed.region_sizes[i] <= compressed.data.len()
            );
        }
        // Offsets should be monotonically non-decreasing.
        for i in 1..3 {
            assert!(compressed.region_offsets[i] >= compressed.region_offsets[i - 1]);
        }
    }

    #[test]
    fn single_region_offset_zero() {
        let w = 4usize;
        let h = 4usize;
        let stride = w * 4;
        let current = make_frame(w, h);
        let previous = vec![0u8; w * h * 4];
        let regions = vec![DamageRect::new(0, 0, w as u32, h as u32)];
        let compressed = compress_frame(&current, &previous, stride, &regions);
        assert_eq!(compressed.region_offsets, vec![0]);
    }

    #[test]
    fn round_trip_multiple_disjoint_regions() {
        let w = 16usize;
        let h = 16usize;
        let stride = w * 4;
        let current: Vec<u8> = (0..w * h * 4).map(|i| (i * 11 + 3) as u8).collect();
        let previous = vec![0u8; w * h * 4];
        let regions = vec![
            DamageRect::new(0, 0, 4, 4),
            DamageRect::new(8, 0, 4, 4),
            DamageRect::new(0, 8, 4, 4),
            DamageRect::new(8, 8, 4, 4),
        ];

        let compressed = compress_frame(&current, &previous, stride, &regions);
        let mut reconstructed = Vec::new();
        decompress_frame(&compressed, &previous, stride, &regions, &mut reconstructed).unwrap();
        // Verify damaged pixels match.
        for rect in &regions {
            for row in 0..rect.height as usize {
                let y = rect.y as usize + row;
                for col in 0..rect.width as usize {
                    let x = rect.x as usize + col;
                    let off = y * stride + x * 4;
                    for i in 0..4 {
                        assert_eq!(
                            reconstructed[off + i],
                            current[off + i],
                            "mismatch at pixel ({x},{y}) byte {i}"
                        );
                    }
                }
            }
        }
    }
}

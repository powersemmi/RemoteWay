//! Unified encode/decode pipeline orchestrating delta + compression stages.

use crate::delta::{DamageRect, delta_decode, delta_encode_simd};
use crate::lz4::{CompressError, compress_region, decompress_region};
use crate::stats::{FrameStats, StageTimer};

/// Output of the compression pipeline for one frame.
#[must_use]
#[derive(Default)]
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

impl CompressedFrame {
    /// Clear all per-frame state, retaining underlying allocations for reuse.
    pub fn reset(&mut self) {
        self.data.clear();
        self.region_offsets.clear();
        self.region_sizes.clear();
        self.stats = FrameStats::default();
    }
}

/// Encode and compress damaged regions of a frame, reusing caller-provided buffers.
///
/// `delta_scratch` and `out` are cleared and refilled in-place; their existing
/// capacity is preserved, so steady-state operation performs no allocations.
///
/// `current` and `previous` must be the same length (width × height × 4 bytes for RGBA).
/// `stride` is the number of bytes per row (`width * 4`).
pub fn compress_frame_into(
    current: &[u8],
    previous: &[u8],
    stride: usize,
    regions: &[DamageRect],
    delta_scratch: &mut Vec<u8>,
    out: &mut CompressedFrame,
) {
    #[cfg(feature = "tracy")]
    let _zone = tracy_client::span!("compress_frame");

    out.reset();
    delta_scratch.clear();

    let encode_timer = StageTimer::start();
    let original_bytes = delta_encode_simd(current, previous, stride, regions, delta_scratch);
    let encode_time = encode_timer.elapsed();

    let compress_timer = StageTimer::start();
    out.region_offsets.reserve(regions.len());
    out.region_sizes.reserve(regions.len());

    // Compress each region independently (could be parallelised via rayon later).
    let mut delta_offset = 0;
    for rect in regions {
        let row_bytes = rect.width as usize * 4;
        let region_len = row_bytes * rect.height as usize;
        let region_delta = &delta_scratch[delta_offset..delta_offset + region_len];
        let compressed = compress_region(region_delta);
        out.region_offsets.push(out.data.len());
        out.region_sizes.push(compressed.len());
        out.data.extend_from_slice(&compressed);
        delta_offset += region_len;
    }
    let compress_time = compress_timer.elapsed();

    out.stats = FrameStats {
        original_bytes,
        compressed_bytes: out.data.len(),
        encode_time,
        compress_time,
    };
}

/// Convenience wrapper: allocate fresh scratch + output buffers and compress.
///
/// Prefer [`compress_frame_into`] in steady-state pipelines to avoid per-frame
/// allocations.
pub fn compress_frame(
    current: &[u8],
    previous: &[u8],
    stride: usize,
    regions: &[DamageRect],
) -> CompressedFrame {
    let mut scratch = Vec::with_capacity(current.len() / 4);
    let mut out = CompressedFrame::default();
    compress_frame_into(current, previous, stride, regions, &mut scratch, &mut out);
    out
}

/// Decompress and reconstruct a frame, reusing caller-provided scratch.
///
/// `delta_scratch` is cleared and refilled; its existing capacity is preserved.
/// `output` is resized to `previous.len()` and filled in-place.
pub fn decompress_frame_into(
    compressed: &CompressedFrame,
    previous: &[u8],
    stride: usize,
    regions: &[DamageRect],
    delta_scratch: &mut Vec<u8>,
    output: &mut Vec<u8>,
) -> Result<(), CompressError> {
    output.resize(previous.len(), 0);

    delta_scratch.clear();
    for (i, _rect) in regions.iter().enumerate() {
        let start = compressed.region_offsets[i];
        let end = start + compressed.region_sizes[i];
        let blob = &compressed.data[start..end];
        let region_delta = decompress_region(blob)?;
        delta_scratch.extend_from_slice(&region_delta);
    }

    delta_decode(delta_scratch, previous, stride, regions, output);
    Ok(())
}

/// Convenience wrapper around [`decompress_frame_into`].
///
/// Prefer the `_into` variant in steady-state pipelines.
pub fn decompress_frame(
    compressed: &CompressedFrame,
    previous: &[u8],
    stride: usize,
    regions: &[DamageRect],
    output: &mut Vec<u8>,
) -> Result<(), CompressError> {
    let mut scratch = Vec::new();
    decompress_frame_into(compressed, previous, stride, regions, &mut scratch, output)
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

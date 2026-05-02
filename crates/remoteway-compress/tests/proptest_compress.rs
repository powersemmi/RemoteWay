use proptest::prelude::*;
use remoteway_compress::delta::{DamageRect, delta_decode, delta_encode_scalar, delta_encode_simd};
use remoteway_compress::pipeline::{compress_frame, decompress_frame};

// ── helpers ──────────────────────────────────────────────────────────────────

const W: usize = 32;
const H: usize = 32;
const STRIDE: usize = W * 4;
const FRAME_BYTES: usize = W * H * 4;

fn arb_frame() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(any::<u8>(), FRAME_BYTES..=FRAME_BYTES)
}

/// Generate a valid non-empty `DamageRect` within the W×H frame.
fn arb_rect() -> impl Strategy<Value = DamageRect> {
    (0u32..W as u32, 0u32..H as u32).prop_flat_map(|(x, y)| {
        let max_w = (W as u32 - x).max(1);
        let max_h = (H as u32 - y).max(1);
        (1u32..=max_w, 1u32..=max_h).prop_map(move |(w, h)| DamageRect::new(x, y, w, h))
    })
}

fn arb_regions(max: usize) -> impl Strategy<Value = Vec<DamageRect>> {
    prop::collection::vec(arb_rect(), 0..=max)
}

// ── proptests ────────────────────────────────────────────────────────────────

proptest! {
    /// delta_encode_scalar followed by delta_decode reconstructs the original pixels.
    #[test]
    fn scalar_round_trip(
        current in arb_frame(),
        previous in arb_frame(),
        regions in arb_regions(8),
    ) {
        let mut delta = Vec::new();
        delta_encode_scalar(&current, &previous, STRIDE, &regions, &mut delta);

        let mut reconstructed = vec![0u8; FRAME_BYTES];
        delta_decode(&delta, &previous, STRIDE, &regions, &mut reconstructed);

        // Each pixel in a damaged region must match `current`.
        for rect in &regions {
            for row in 0..rect.height as usize {
                let y = rect.y as usize + row;
                for col in 0..rect.width as usize {
                    let x = rect.x as usize + col;
                    let off = y * STRIDE + x * 4;
                    prop_assert_eq!(
                        &reconstructed[off..off + 4],
                        &current[off..off + 4],
                        "pixel ({},{}) mismatch", x, y
                    );
                }
            }
        }
    }

    /// SIMD path produces byte-identical output to the scalar path.
    #[test]
    fn simd_matches_scalar(
        current in arb_frame(),
        previous in arb_frame(),
        regions in arb_regions(8),
    ) {
        let mut out_scalar = Vec::new();
        let mut out_simd = Vec::new();
        delta_encode_scalar(&current, &previous, STRIDE, &regions, &mut out_scalar);
        delta_encode_simd(&current, &previous, STRIDE, &regions, &mut out_simd);
        prop_assert_eq!(out_scalar, out_simd);
    }

    /// compress_frame + decompress_frame reconstructs damaged pixels exactly.
    #[test]
    fn pipeline_round_trip(
        current in arb_frame(),
        previous in arb_frame(),
        regions in arb_regions(4),
    ) {
        let compressed = compress_frame(&current, &previous, STRIDE, &regions);
        let mut reconstructed = Vec::new();
        decompress_frame(&compressed, &previous, STRIDE, &regions, &mut reconstructed).unwrap();

        for rect in &regions {
            for row in 0..rect.height as usize {
                let y = rect.y as usize + row;
                for col in 0..rect.width as usize {
                    let x = rect.x as usize + col;
                    let off = y * STRIDE + x * 4;
                    prop_assert_eq!(
                        &reconstructed[off..off + 4],
                        &current[off..off + 4],
                        "pixel ({},{}) mismatch after pipeline round-trip", x, y
                    );
                }
            }
        }
    }

    /// Empty regions always produce zero-length delta and stats.
    #[test]
    fn empty_regions_zero_output(
        current in arb_frame(),
        previous in arb_frame(),
    ) {
        let mut out = Vec::new();
        let n = delta_encode_scalar(&current, &previous, STRIDE, &[], &mut out);
        prop_assert_eq!(n, 0usize);
        prop_assert!(out.is_empty());
    }

    /// Identical current/previous always produces an all-zero delta.
    #[test]
    fn unchanged_frame_all_zero_delta(
        frame in arb_frame(),
        regions in arb_regions(8),
    ) {
        let mut out = Vec::new();
        delta_encode_scalar(&frame, &frame, STRIDE, &regions, &mut out);
        prop_assert!(out.iter().all(|&b| b == 0), "unchanged frame must produce all-zero delta");
    }
}

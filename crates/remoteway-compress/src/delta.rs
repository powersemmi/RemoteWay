use wide::u8x32;

/// A rectangular region of a frame that changed since the previous frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DamageRect {
    /// Left edge of the damaged region, in pixels.
    pub x: u32,
    /// Top edge of the damaged region, in pixels.
    pub y: u32,
    /// Width of the damaged region, in pixels.
    pub width: u32,
    /// Height of the damaged region, in pixels.
    pub height: u32,
}

impl DamageRect {
    /// Create a new damage rectangle.
    #[must_use]
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Number of pixels (RGBA quads) covered by this rect.
    #[must_use]
    pub fn pixel_count(&self) -> usize {
        (self.width as usize) * (self.height as usize)
    }
}

/// XOR delta encode `current` against `previous`, writing only the bytes at
/// positions covered by `regions` into `output`.
///
/// Bytes that are already zero (unchanged) are still written — the compressor
/// downstream handles zero runs efficiently. This avoids a branch on the hot path.
///
/// Returns the number of bytes written into `output`.
pub fn delta_encode_scalar(
    current: &[u8],
    previous: &[u8],
    stride: usize,
    regions: &[DamageRect],
    output: &mut Vec<u8>,
) -> usize {
    debug_assert_eq!(current.len(), previous.len());
    let before = output.len();
    for rect in regions {
        for row in 0..rect.height as usize {
            let y = rect.y as usize + row;
            let x_start = rect.x as usize * 4; // RGBA: 4 bytes/pixel
            let x_end = x_start + rect.width as usize * 4;
            let row_start = y * stride + x_start;
            let row_end = y * stride + x_end;
            let cur = &current[row_start..row_end];
            let prev = &previous[row_start..row_end];
            output.extend(cur.iter().zip(prev.iter()).map(|(c, p)| c ^ p));
        }
    }
    output.len() - before
}

/// SIMD-accelerated delta encode using `wide::u8x32` (256-bit lanes).
///
/// Falls back to scalar for the tail when the chunk length is not a multiple of 32.
/// Output is byte-for-byte identical to [`delta_encode_scalar`].
pub fn delta_encode_simd(
    current: &[u8],
    previous: &[u8],
    stride: usize,
    regions: &[DamageRect],
    output: &mut Vec<u8>,
) -> usize {
    debug_assert_eq!(current.len(), previous.len());
    let before = output.len();

    for rect in regions {
        for row in 0..rect.height as usize {
            let y = rect.y as usize + row;
            let x_start = rect.x as usize * 4;
            let x_end = x_start + rect.width as usize * 4;
            let row_start = y * stride + x_start;
            let row_end = y * stride + x_end;
            let cur = &current[row_start..row_end];
            let prev = &previous[row_start..row_end];
            xor_into_vec_simd(cur, prev, output);
        }
    }

    output.len() - before
}

/// XOR two equal-length slices into `dst` using u8x32 SIMD + scalar tail.
fn xor_into_vec_simd(a: &[u8], b: &[u8], dst: &mut Vec<u8>) {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let chunks = len / 32;
    let tail = len % 32;

    dst.reserve(len);

    for i in 0..chunks {
        let off = i * 32;
        // SAFETY: off+32 ≤ len (checked by chunks = len/32), slices are valid.
        let va = u8x32::new(*array_ref!(a, off, 32));
        let vb = u8x32::new(*array_ref!(b, off, 32));
        let result: [u8; 32] = (va ^ vb).into();
        dst.extend_from_slice(&result);
    }

    let tail_off = chunks * 32;
    for i in 0..tail {
        dst.push(a[tail_off + i] ^ b[tail_off + i]);
    }
}

/// Helper macro for `array_ref` slices (avoids unsafe indexing boilerplate).
macro_rules! array_ref {
    ($slice:expr, $offset:expr, $len:expr) => {{
        let slice: &[u8] = &$slice[$offset..$offset + $len];
        // SAFETY: We verified the slice has exactly $len elements.
        unsafe { &*(slice.as_ptr() as *const [u8; $len]) }
    }};
}

use array_ref;

/// Reconstruct the delta-encoded output back into `output` by XOR-ing with `previous`.
pub fn delta_decode(
    delta: &[u8],
    previous: &[u8],
    stride: usize,
    regions: &[DamageRect],
    output: &mut [u8],
) {
    output.copy_from_slice(previous);
    let mut offset = 0;
    for rect in regions {
        for row in 0..rect.height as usize {
            let y = rect.y as usize + row;
            let x_start = rect.x as usize * 4;
            let x_end = x_start + rect.width as usize * 4;
            let row_start = y * stride + x_start;
            let row_len = x_end - x_start;
            for i in 0..row_len {
                output[row_start + i] = previous[row_start + i] ^ delta[offset + i];
            }
            offset += row_len;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(w: usize, h: usize, fill: u8) -> Vec<u8> {
        vec![fill; w * h * 4]
    }

    fn full_frame_region(w: u32, h: u32) -> Vec<DamageRect> {
        vec![DamageRect::new(0, 0, w, h)]
    }

    #[test]
    fn simd_matches_scalar_random() {
        let w = 64usize;
        let h = 32usize;
        let stride = w * 4;
        let current: Vec<u8> = (0..w * h * 4).map(|i| (i * 7 + 13) as u8).collect();
        let previous: Vec<u8> = (0..w * h * 4).map(|i| (i * 3 + 5) as u8).collect();
        let regions = full_frame_region(w as u32, h as u32);

        let mut out_scalar = Vec::new();
        let mut out_simd = Vec::new();
        delta_encode_scalar(&current, &previous, stride, &regions, &mut out_scalar);
        delta_encode_simd(&current, &previous, stride, &regions, &mut out_simd);
        assert_eq!(
            out_scalar, out_simd,
            "SIMD and scalar delta must produce identical output"
        );
    }

    #[test]
    fn round_trip_encode_decode() {
        let w = 8usize;
        let h = 4usize;
        let stride = w * 4;
        let current: Vec<u8> = (0..w * h * 4).map(|i| i as u8).collect();
        let previous = make_frame(w, h, 0);
        let regions = full_frame_region(w as u32, h as u32);

        let mut delta = Vec::new();
        delta_encode_scalar(&current, &previous, stride, &regions, &mut delta);

        let mut reconstructed = vec![0u8; w * h * 4];
        delta_decode(&delta, &previous, stride, &regions, &mut reconstructed);
        assert_eq!(reconstructed, current);
    }

    #[test]
    fn empty_damage_regions_zero_output() {
        let w = 16usize;
        let h = 16usize;
        let stride = w * 4;
        let current = make_frame(w, h, 0xFF);
        let previous = make_frame(w, h, 0x00);
        let mut out = Vec::new();
        let n = delta_encode_scalar(&current, &previous, stride, &[], &mut out);
        assert_eq!(n, 0);
        assert!(out.is_empty());
    }

    #[test]
    fn unchanged_frame_produces_all_zeros() {
        let w = 4usize;
        let h = 4usize;
        let stride = w * 4;
        let frame = make_frame(w, h, 0xAB);
        let regions = full_frame_region(w as u32, h as u32);
        let mut out = Vec::new();
        delta_encode_scalar(&frame, &frame, stride, &regions, &mut out);
        assert!(out.iter().all(|&b| b == 0));
    }

    #[test]
    fn partial_damage_rect() {
        let w = 8usize;
        let h = 8usize;
        let stride = w * 4;
        let mut current = make_frame(w, h, 0);
        let previous = make_frame(w, h, 0);
        // Change a 2x2 region at (2,2).
        for y in 2..4 {
            for x in 2..4 {
                let off = y * stride + x * 4;
                current[off] = 0xFF;
            }
        }
        let regions = vec![DamageRect::new(2, 2, 2, 2)];
        let mut out_scalar = Vec::new();
        let mut out_simd = Vec::new();
        delta_encode_scalar(&current, &previous, stride, &regions, &mut out_scalar);
        delta_encode_simd(&current, &previous, stride, &regions, &mut out_simd);
        assert_eq!(out_scalar, out_simd);
        // 2x2 pixels × 4 bytes = 16 bytes output.
        assert_eq!(out_scalar.len(), 16);
    }

    #[test]
    fn non_multiple_of_32_simd() {
        let w = 5usize;
        let h = 3usize;
        let stride = w * 4;
        let current: Vec<u8> = (0..w * h * 4).map(|i| i as u8).collect();
        let previous: Vec<u8> = (0..w * h * 4).map(|i| (i + 1) as u8).collect();
        let regions = full_frame_region(w as u32, h as u32);
        let mut out_s = Vec::new();
        let mut out_v = Vec::new();
        delta_encode_scalar(&current, &previous, stride, &regions, &mut out_s);
        delta_encode_simd(&current, &previous, stride, &regions, &mut out_v);
        assert_eq!(out_s, out_v);
    }

    #[test]
    fn damage_rect_pixel_count() {
        assert_eq!(DamageRect::new(0, 0, 10, 20).pixel_count(), 200);
    }

    #[test]
    fn damage_rect_pixel_count_zero_width() {
        assert_eq!(DamageRect::new(0, 0, 0, 5).pixel_count(), 0);
    }

    #[test]
    fn damage_rect_pixel_count_zero_height() {
        assert_eq!(DamageRect::new(0, 0, 5, 0).pixel_count(), 0);
    }

    #[test]
    fn damage_rect_fields() {
        let r = DamageRect::new(3, 7, 11, 13);
        assert_eq!(r.x, 3);
        assert_eq!(r.y, 7);
        assert_eq!(r.width, 11);
        assert_eq!(r.height, 13);
    }

    #[test]
    fn damage_rect_debug_clone_copy_eq() {
        let r = DamageRect::new(1, 2, 3, 4);
        let r2 = r; // Copy
        assert_eq!(r, r2);
        let r3 = r;
        assert_eq!(r, r3);
        let dbg = format!("{:?}", r);
        assert!(dbg.contains("DamageRect"));
    }

    #[test]
    fn one_by_one_rect_encode_decode() {
        let w = 4usize;
        let h = 4usize;
        let stride = w * 4;
        let current: Vec<u8> = (0..w * h * 4).map(|i| (i + 1) as u8).collect();
        let previous = make_frame(w, h, 0);
        let regions = vec![DamageRect::new(1, 1, 1, 1)];

        let mut delta = Vec::new();
        delta_encode_scalar(&current, &previous, stride, &regions, &mut delta);
        assert_eq!(delta.len(), 4); // 1 pixel × 4 bytes

        let mut out = vec![0u8; w * h * 4];
        delta_decode(&delta, &previous, stride, &regions, &mut out);
        // Only the pixel at (1,1) should be non-zero.
        let pixel_off = stride + 4;
        for i in 0..4 {
            assert_eq!(out[pixel_off + i], current[pixel_off + i]);
        }
    }

    #[test]
    fn one_by_two_rect() {
        let w = 4usize;
        let h = 4usize;
        let stride = w * 4;
        let current: Vec<u8> = (0..w * h * 4).map(|i| i as u8).collect();
        let previous = make_frame(w, h, 0);
        let regions = vec![DamageRect::new(0, 0, 1, 2)]; // 1 wide, 2 tall

        let mut delta_s = Vec::new();
        let mut delta_v = Vec::new();
        delta_encode_scalar(&current, &previous, stride, &regions, &mut delta_s);
        delta_encode_simd(&current, &previous, stride, &regions, &mut delta_v);
        assert_eq!(delta_s, delta_v);
        assert_eq!(delta_s.len(), 2 * 4); // 1×2 pixels
    }

    #[test]
    fn stride_with_padding() {
        let w = 3usize;
        let h = 2usize;
        // Extra padding: stride > w*4.
        let stride = w * 4 + 64;
        let current: Vec<u8> = (0..stride * h).map(|i| (i * 5) as u8).collect();
        let previous: Vec<u8> = (0..stride * h).map(|i| (i * 3) as u8).collect();
        let regions = full_frame_region(w as u32, h as u32);

        let mut delta_s = Vec::new();
        let mut delta_v = Vec::new();
        delta_encode_scalar(&current, &previous, stride, &regions, &mut delta_s);
        delta_encode_simd(&current, &previous, stride, &regions, &mut delta_v);
        assert_eq!(delta_s, delta_v);

        let mut reconstructed = vec![0u8; stride * h];
        delta_decode(&delta_s, &previous, stride, &regions, &mut reconstructed);
        // Only the pixel area should be reconstructed; padding is from previous.
        for row in 0..h {
            for col in 0..w {
                let off = row * stride + col * 4;
                for i in 0..4 {
                    assert_eq!(reconstructed[off + i], current[off + i]);
                }
            }
        }
    }

    /// Overlapping damage rects must not cause UB, panics, or out-of-bounds accesses.
    /// The encode is applied per-rect; decode last-writer-wins for the overlap area —
    /// behaviour is defined, just not semantically meaningful for overlapping damage.
    #[test]
    fn overlapping_damage_regions_no_ub() {
        let w = 8usize;
        let h = 8usize;
        let stride = w * 4;
        let current: Vec<u8> = (0..w * h * 4).map(|i| (i * 3 + 7) as u8).collect();
        let previous: Vec<u8> = (0..w * h * 4).map(|i| (i * 5 + 2) as u8).collect();

        // Two rects that fully overlap.
        let rect_a = DamageRect::new(2, 2, 4, 4);
        let rect_b = DamageRect::new(2, 2, 4, 4);
        let regions = vec![rect_a, rect_b];

        // Must not panic.
        let mut out_scalar = Vec::new();
        let n_scalar = delta_encode_scalar(&current, &previous, stride, &regions, &mut out_scalar);
        assert_eq!(n_scalar, out_scalar.len());

        let mut out_simd = Vec::new();
        let n_simd = delta_encode_simd(&current, &previous, stride, &regions, &mut out_simd);
        assert_eq!(n_simd, out_simd.len());

        // Both paths produce identical output.
        assert_eq!(out_scalar, out_simd);

        // Output size = 2 rects × 4×4 pixels × 4 bytes.
        assert_eq!(out_scalar.len(), 2 * 4 * 4 * 4);

        // Decode must not panic either.
        let mut reconstructed = vec![0u8; w * h * 4];
        delta_decode(&out_scalar, &previous, stride, &regions, &mut reconstructed);
        // No assertion on value — overlapping decode result is defined but not meaningful.
        // The key property: no panic, no out-of-bounds.
    }
}

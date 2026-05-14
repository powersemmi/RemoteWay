//! Minimal on-frame FPS overlay used by the `--debug` flag.
//!
//! Draws a small "FPS: 60.0" text in the top-left corner of an RGBA8 frame
//! buffer. Uses a hand-rolled 5×7 monospace bitmap font for the characters
//! actually needed (digits, '.', ':', 'F', 'P', 'S', ' ') to avoid pulling
//! in a real font/text-rendering dependency for one overlay.

/// Glyph dimensions: each character is 5 columns × 7 rows of bits.
const GLYPH_W: usize = 5;
const GLYPH_H: usize = 7;

/// Bitmap for one glyph: 7 rows, low 5 bits of each `u8` are the pixels
/// (bit 4 = leftmost column, bit 0 = rightmost). `None` = missing glyph.
fn glyph(c: char) -> Option<[u8; GLYPH_H]> {
    match c {
        '0' => Some([0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110]),
        '1' => Some([0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110]),
        '2' => Some([0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111]),
        '3' => Some([0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110]),
        '4' => Some([0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010]),
        '5' => Some([0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110]),
        '6' => Some([0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110]),
        '7' => Some([0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000]),
        '8' => Some([0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110]),
        '9' => Some([0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100]),
        '.' => Some([0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00100]),
        ':' => Some([0b00000, 0b00100, 0b00100, 0b00000, 0b00100, 0b00100, 0b00000]),
        'F' => Some([0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000]),
        'P' => Some([0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000]),
        'S' => Some([0b01110, 0b10001, 0b10000, 0b01110, 0b00001, 0b10001, 0b01110]),
        ' ' => Some([0; GLYPH_H]),
        _ => None,
    }
}

/// One transparent-pixel padding column to the right of each glyph.
const GLYPH_STRIDE: usize = GLYPH_W + 1;

/// Per-pixel scale factor — at scale=3 each glyph cell is 15×21 px.
pub const TEXT_SCALE: usize = 3;

/// Solid black padding around the text, in pixels.
const PAD: usize = 4;

/// Write a single RGBA8 pixel at `(x, y)` if it falls inside the buffer.
fn put_pixel(
    buf: &mut [u8],
    width: u32,
    height: u32,
    stride: u32,
    x: i32,
    y: i32,
    rgba: [u8; 4],
) {
    if x < 0 || y < 0 {
        return;
    }
    let (x, y) = (x as u32, y as u32);
    if x >= width || y >= height {
        return;
    }
    let off = (y * stride + x * 4) as usize;
    if off + 4 <= buf.len() {
        buf[off..off + 4].copy_from_slice(&rgba);
    }
}

/// Render `text` at pixel `(x, y)` on an RGBA8 buffer of size
/// `width × height` with `stride` bytes per row. Draws a semi-opaque black
/// background rectangle behind the text for legibility on any content.
fn draw_text(
    buf: &mut [u8],
    width: u32,
    height: u32,
    stride: u32,
    x: i32,
    y: i32,
    text: &str,
) {
    let text_px_w = text.chars().count() * GLYPH_STRIDE * TEXT_SCALE;
    let text_px_h = GLYPH_H * TEXT_SCALE;
    let bg_w = text_px_w as i32 + PAD as i32 * 2;
    let bg_h = text_px_h as i32 + PAD as i32 * 2;

    // Background: opaque black so the white text is readable on any frame.
    for by in 0..bg_h {
        for bx in 0..bg_w {
            put_pixel(buf, width, height, stride, x + bx, y + by, [0, 0, 0, 255]);
        }
    }

    // Foreground: white, integer up-scaled by TEXT_SCALE.
    let text_x = x + PAD as i32;
    let text_y = y + PAD as i32;
    for (i, ch) in text.chars().enumerate() {
        let rows = match glyph(ch) {
            Some(r) => r,
            None => continue,
        };
        let cx = text_x + (i * GLYPH_STRIDE * TEXT_SCALE) as i32;
        for (row_idx, row_bits) in rows.iter().enumerate() {
            for col in 0..GLYPH_W {
                let bit = (row_bits >> (GLYPH_W - 1 - col)) & 1;
                if bit == 0 {
                    continue;
                }
                let px_x = cx + (col * TEXT_SCALE) as i32;
                let px_y = text_y + (row_idx * TEXT_SCALE) as i32;
                for dy in 0..TEXT_SCALE as i32 {
                    for dx in 0..TEXT_SCALE as i32 {
                        put_pixel(
                            buf,
                            width,
                            height,
                            stride,
                            px_x + dx,
                            px_y + dy,
                            [255, 255, 255, 255],
                        );
                    }
                }
            }
        }
    }
}

/// The bounding rectangle of the FPS overlay in pixels, at the current
/// font scale and padding. Used to extend the damage list so the overlay
/// is included in partial commits to the compositor.
pub fn overlay_rect(x: u32, y: u32) -> (u32, u32, u32, u32) {
    // Worst-case glyph count: "FPS: 999.9" is 10 characters.
    let chars: usize = 10;
    let w = (chars * GLYPH_STRIDE * TEXT_SCALE + PAD * 2) as u32;
    let h = (GLYPH_H * TEXT_SCALE + PAD * 2) as u32;
    (x, y, w, h)
}

/// Draw the FPS readout in the top-left corner of `buf`.
pub fn draw_fps(buf: &mut [u8], width: u32, height: u32, stride: u32, fps: f32) {
    // Clamp + format with one decimal, e.g. "FPS: 59.7". Negative or NaN
    // values get pinned to 0 so the format width stays predictable.
    let clamped = if fps.is_finite() && fps >= 0.0 {
        fps.min(999.9)
    } else {
        0.0
    };
    let text = format!("FPS: {:>5.1}", clamped);
    draw_text(buf, width, height, stride, 10, 10, &text);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glyph_table_covers_required_characters() {
        for ch in "0123456789FPS.: ".chars() {
            assert!(glyph(ch).is_some(), "missing glyph for {ch:?}");
        }
    }

    #[test]
    fn draw_text_does_not_panic_on_tiny_buffer() {
        // Buffer smaller than the overlay must be clipped, not crashed.
        let w = 16u32;
        let h = 16u32;
        let mut buf = vec![0u8; (w * h * 4) as usize];
        draw_text(&mut buf, w, h, w * 4, 0, 0, "FPS: 60.0");
        // The top-left corner pixel must have been written (background fill).
        assert_eq!(buf[3], 255, "alpha not set on top-left pixel");
    }

    #[test]
    fn draw_fps_clamps_garbage_values() {
        let mut buf = vec![0u8; 256 * 256 * 4];
        // None of these should panic.
        draw_fps(&mut buf, 256, 256, 256 * 4, f32::NAN);
        draw_fps(&mut buf, 256, 256, 256 * 4, f32::INFINITY);
        draw_fps(&mut buf, 256, 256, 256 * 4, -42.0);
        draw_fps(&mut buf, 256, 256, 256 * 4, 99999.0);
    }

    #[test]
    fn overlay_rect_is_positive() {
        let (x, y, w, h) = overlay_rect(10, 10);
        assert_eq!((x, y), (10, 10));
        assert!(w > 0 && h > 0);
    }
}

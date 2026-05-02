//! Cursor position updates and cursor-image metadata.

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

/// Cursor position and hotspot update (no bitmap — just movement).
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct CursorMove {
    /// Target surface ID.
    pub surface_id: u16,
    /// Alignment padding; always zero.
    pub _pad: u16,
    /// X position in surface-local coordinates.
    pub x: f32,
    /// Y position in surface-local coordinates.
    pub y: f32,
}

const _: () = assert!(size_of::<CursorMove>() == 12);

/// Full cursor update: position + hotspot + optional RGBA bitmap dimensions.
/// The bitmap pixel data follows this header in the payload when `has_bitmap != 0`.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct CursorUpdate {
    /// Target surface ID.
    pub surface_id: u16,
    /// Hotspot X offset within the cursor image.
    pub hotspot_x: i16,
    /// Hotspot Y offset within the cursor image.
    pub hotspot_y: i16,
    /// 1 if RGBA bitmap data follows in the payload.
    pub has_bitmap: u8,
    /// Alignment padding; always zero.
    pub _pad: u8,
    /// Width of the cursor bitmap in pixels.
    pub bitmap_width: u16,
    /// Height of the cursor bitmap in pixels.
    pub bitmap_height: u16,
    /// X position in surface-local coordinates.
    pub x: f32,
    /// Y position in surface-local coordinates.
    pub y: f32,
}

const _: () = assert!(size_of::<CursorUpdate>() == 20);

#[cfg(test)]
mod tests {
    use zerocopy::IntoBytes;

    use super::*;

    #[test]
    fn cursor_move_size() {
        assert_eq!(size_of::<CursorMove>(), 12);
    }

    #[test]
    fn cursor_update_size() {
        assert_eq!(size_of::<CursorUpdate>(), 20);
    }

    #[test]
    fn cursor_move_round_trip() {
        let mv = CursorMove {
            surface_id: 1,
            _pad: 0,
            x: 100.0,
            y: 200.0,
        };
        let bytes = mv.as_bytes();
        let decoded = CursorMove::ref_from_bytes(bytes).unwrap();
        assert_eq!({ decoded.surface_id }, 1);
        assert_eq!({ decoded.x }, 100.0);
    }

    #[test]
    fn cursor_update_round_trip() {
        let cu = CursorUpdate {
            surface_id: 5,
            hotspot_x: -3,
            hotspot_y: 7,
            has_bitmap: 1,
            _pad: 0,
            bitmap_width: 32,
            bitmap_height: 32,
            x: 100.5,
            y: 200.0,
        };
        let bytes = cu.as_bytes();
        let decoded = CursorUpdate::ref_from_bytes(bytes).unwrap();
        assert_eq!({ decoded.surface_id }, 5);
        assert_eq!({ decoded.hotspot_x }, -3);
        assert_eq!({ decoded.hotspot_y }, 7);
        assert_eq!({ decoded.has_bitmap }, 1);
        assert_eq!({ decoded.bitmap_width }, 32);
        assert_eq!({ decoded.bitmap_height }, 32);
        assert_eq!({ decoded.x }, 100.5);
        assert_eq!({ decoded.y }, 200.0);
    }

    #[test]
    fn cursor_update_no_bitmap() {
        let cu = CursorUpdate {
            surface_id: 1,
            hotspot_x: 0,
            hotspot_y: 0,
            has_bitmap: 0,
            _pad: 0,
            bitmap_width: 0,
            bitmap_height: 0,
            x: 50.0,
            y: 60.0,
        };
        let bytes = cu.as_bytes();
        let decoded = CursorUpdate::ref_from_bytes(bytes).unwrap();
        assert_eq!({ decoded.has_bitmap }, 0);
        assert_eq!({ decoded.bitmap_width }, 0);
    }

    #[test]
    fn cursor_move_negative_coords() {
        let mv = CursorMove {
            surface_id: 0,
            _pad: 0,
            x: -50.0,
            y: -100.5,
        };
        let bytes = mv.as_bytes();
        let decoded = CursorMove::ref_from_bytes(bytes).unwrap();
        assert_eq!({ decoded.x }, -50.0);
        assert_eq!({ decoded.y }, -100.5);
    }

    #[test]
    fn cursor_update_negative_hotspot() {
        let cu = CursorUpdate {
            surface_id: 0,
            hotspot_x: -10,
            hotspot_y: -20,
            has_bitmap: 0,
            _pad: 0,
            bitmap_width: 0,
            bitmap_height: 0,
            x: 0.0,
            y: 0.0,
        };
        let bytes = cu.as_bytes();
        let decoded = CursorUpdate::ref_from_bytes(bytes).unwrap();
        assert_eq!({ decoded.hotspot_x }, -10);
        assert_eq!({ decoded.hotspot_y }, -20);
    }
}

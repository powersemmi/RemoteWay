//! Frame metadata, damage regions, and wire-area descriptors.

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

/// Frame payload wire header: dimensions + number of damage regions.
///
/// Sent at the start of every `FrameUpdate` / `AnchorFrame` payload,
/// followed by `num_regions` [`WireRegion`] descriptors and the compressed data.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct FrameMeta {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Row stride in bytes.
    pub stride: u32,
    /// Number of [`WireRegion`] descriptors following this header.
    pub num_regions: u32,
}

impl FrameMeta {
    /// Size of `FrameMeta` in bytes (always 16).
    pub const SIZE: usize = size_of::<Self>();

    /// Create a new `FrameMeta` with the given dimensions.
    #[must_use]
    pub fn new(width: u32, height: u32, stride: u32, num_regions: u32) -> Self {
        Self {
            width,
            height,
            stride,
            num_regions,
        }
    }
}

/// Per-region descriptor: rectangle coordinates + compressed blob size.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct WireRegion {
    /// Left edge of the dirty rectangle, in pixels.
    pub x: u32,
    /// Top edge of the dirty rectangle, in pixels.
    pub y: u32,
    /// Width of the dirty rectangle, in pixels.
    pub w: u32,
    /// Height of the dirty rectangle, in pixels.
    pub h: u32,
    /// Size of the compressed payload for this region.
    pub compressed_size: u32,
}

impl WireRegion {
    /// Size of `WireRegion` in bytes (always 20).
    pub const SIZE: usize = size_of::<Self>();

    /// Create a new `WireRegion` descriptor.
    #[must_use]
    pub fn new(x: u32, y: u32, w: u32, h: u32, compressed_size: u32) -> Self {
        Self {
            x,
            y,
            w,
            h,
            compressed_size,
        }
    }
}

const _: () = assert!(size_of::<FrameMeta>() == 16);
const _: () = assert!(size_of::<WireRegion>() == 20);

#[cfg(test)]
mod tests {
    use zerocopy::{FromBytes, IntoBytes};

    use super::*;

    #[test]
    fn frame_meta_size() {
        assert_eq!(FrameMeta::SIZE, 16);
    }

    #[test]
    fn wire_region_size() {
        assert_eq!(WireRegion::SIZE, 20);
    }

    #[test]
    fn frame_meta_round_trip() {
        let meta = FrameMeta::new(1920, 1080, 7680, 3);
        let bytes = meta.as_bytes();
        let decoded = FrameMeta::ref_from_bytes(bytes).unwrap();
        assert_eq!({ decoded.width }, 1920);
        assert_eq!({ decoded.height }, 1080);
        assert_eq!({ decoded.stride }, 7680);
        assert_eq!({ decoded.num_regions }, 3);
    }

    #[test]
    fn wire_region_round_trip() {
        let wr = WireRegion::new(10, 20, 100, 50, 4096);
        let bytes = wr.as_bytes();
        let decoded = WireRegion::ref_from_bytes(bytes).unwrap();
        assert_eq!({ decoded.x }, 10);
        assert_eq!({ decoded.y }, 20);
        assert_eq!({ decoded.w }, 100);
        assert_eq!({ decoded.h }, 50);
        assert_eq!({ decoded.compressed_size }, 4096);
    }

    #[test]
    fn serialize_deserialize_sequence() {
        let meta = FrameMeta::new(3840, 2160, 15360, 2);
        let r0 = WireRegion::new(0, 0, 1920, 1080, 512);
        let r1 = WireRegion::new(1920, 0, 1920, 1080, 1024);

        let mut buf = Vec::new();
        buf.extend_from_slice(meta.as_bytes());
        buf.extend_from_slice(r0.as_bytes());
        buf.extend_from_slice(r1.as_bytes());

        assert_eq!(buf.len(), FrameMeta::SIZE + 2 * WireRegion::SIZE);

        let parsed_meta = FrameMeta::ref_from_bytes(&buf[..FrameMeta::SIZE]).unwrap();
        assert_eq!({ parsed_meta.num_regions }, 2);

        let off1 = FrameMeta::SIZE;
        let pr0 = WireRegion::ref_from_bytes(&buf[off1..off1 + WireRegion::SIZE]).unwrap();
        assert_eq!({ pr0.compressed_size }, 512);

        let off2 = off1 + WireRegion::SIZE;
        let pr1 = WireRegion::ref_from_bytes(&buf[off2..off2 + WireRegion::SIZE]).unwrap();
        assert_eq!({ pr1.compressed_size }, 1024);
    }
}

//! Client-requested downscaling for bandwidth-adaptive streaming.

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

/// Client → server request to produce frames at the given resolution.
///
/// Sent after the handshake. The server downscales captured frames to
/// `width × height` before delta encoding and compression, significantly
/// reducing bandwidth when the client doesn't need native resolution.
///
/// A payload with `width = 0` and `height = 0` resets to native resolution.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct TargetResolutionPayload {
    /// Requested frame width in pixels (0 = native).
    pub width: u32,
    /// Requested frame height in pixels (0 = native).
    pub height: u32,
}

const _: () = assert!(
    size_of::<TargetResolutionPayload>() == 8,
    "TargetResolutionPayload must be exactly 8 bytes"
);

impl TargetResolutionPayload {
    /// Size of `TargetResolutionPayload` in bytes (always 8).
    pub const SIZE: usize = 8;

    /// Create a new target resolution request.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

#[cfg(test)]
mod tests {
    use zerocopy::{FromBytes, IntoBytes};

    use super::*;

    #[test]
    fn size_is_8() {
        assert_eq!(size_of::<TargetResolutionPayload>(), 8);
        assert_eq!(TargetResolutionPayload::SIZE, 8);
    }

    #[test]
    fn round_trip() {
        let p = TargetResolutionPayload::new(1920, 1080);
        let bytes = p.as_bytes();
        let decoded = TargetResolutionPayload::ref_from_bytes(bytes).unwrap();
        assert_eq!({ decoded.width }, 1920);
        assert_eq!({ decoded.height }, 1080);
    }

    #[test]
    fn zero_means_native() {
        let p = TargetResolutionPayload::new(0, 0);
        assert_eq!({ p.width }, 0);
        assert_eq!({ p.height }, 0);
    }
}

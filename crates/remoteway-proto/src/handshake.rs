//! Connection handshake with capability negotiation.

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

/// Capture backend flags (bitmask).
pub mod capture_flags {
    /// wlr-screencopy protocol (wlroots).
    pub const WLR_SCREENCOPY: u8 = 1 << 0;
    /// ext-image-capture-source-v1 protocol.
    pub const EXT_IMAGE_CAPTURE: u8 = 1 << 1;
    /// XDG Desktop Portal screen-cast.
    pub const PORTAL: u8 = 1 << 2;
}

/// Compression algorithm flags (bitmask).
pub mod compress_flags {
    /// LZ4 fast compression.
    pub const LZ4: u8 = 1 << 0;
    /// Zstandard compression (higher ratio).
    pub const ZSTD: u8 = 1 << 1;
}

/// Fixed-size handshake payload exchanged at connection start.
///
/// Client and server each send one; the intersection of their flags
/// determines the negotiated protocol.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct HandshakePayload {
    /// Protocol version — must match on both sides.
    pub version: u16,
    /// Bitmask of supported capture backends (server → client).
    pub capture_flags: u8,
    /// Bitmask of supported compression algorithms.
    pub compress_flags: u8,
    /// Reserved for future use; zeroed on send, ignored on receive.
    pub _reserved: [u8; 4],
}

const _: () = assert!(size_of::<HandshakePayload>() == 8);

impl HandshakePayload {
    /// Current protocol version (incremented on breaking changes).
    pub const PROTOCOL_VERSION: u16 = 1;

    /// Create a new handshake payload with the given capability flags.
    #[must_use]
    pub fn new(capture_flags: u8, compress_flags: u8) -> Self {
        Self {
            version: Self::PROTOCOL_VERSION,
            capture_flags,
            compress_flags,
            _reserved: [0; 4],
        }
    }
}

#[cfg(test)]
mod tests {
    use zerocopy::IntoBytes;

    use super::*;

    #[test]
    fn handshake_is_8_bytes() {
        assert_eq!(size_of::<HandshakePayload>(), 8);
    }

    #[test]
    fn handshake_round_trip() {
        let hs = HandshakePayload::new(
            capture_flags::WLR_SCREENCOPY | capture_flags::EXT_IMAGE_CAPTURE,
            compress_flags::LZ4,
        );
        let bytes = hs.as_bytes();
        let decoded = HandshakePayload::ref_from_bytes(bytes).unwrap();
        assert_eq!({ decoded.version }, HandshakePayload::PROTOCOL_VERSION);
        assert_ne!(decoded.capture_flags & capture_flags::WLR_SCREENCOPY, 0);
        assert_ne!(decoded.compress_flags & compress_flags::LZ4, 0);
    }

    #[test]
    fn capture_flags_no_overlap() {
        assert_eq!(
            capture_flags::WLR_SCREENCOPY & capture_flags::EXT_IMAGE_CAPTURE,
            0
        );
    }

    #[test]
    fn compress_flags_no_overlap() {
        assert_eq!(compress_flags::LZ4 & compress_flags::ZSTD, 0);
    }

    #[test]
    fn protocol_version_is_one() {
        assert_eq!(HandshakePayload::PROTOCOL_VERSION, 1);
    }

    #[test]
    fn new_sets_version_automatically() {
        let hs = HandshakePayload::new(0, 0);
        assert_eq!({ hs.version }, HandshakePayload::PROTOCOL_VERSION);
    }

    #[test]
    fn reserved_field_zeroed() {
        let hs = HandshakePayload::new(0xFF, 0xFF);
        assert_eq!(hs._reserved, [0; 4]);
    }
}

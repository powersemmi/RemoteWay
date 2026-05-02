use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

/// Clipboard transfer header (8 bytes), followed by variable-length data.
///
/// The clipboard data follows immediately after this header in the payload.
/// `data_len` bytes of clipboard content in the specified MIME type.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct ClipboardHeader {
    /// MIME type discriminant.
    pub mime_type: u8,
    /// Direction: 0 = server→client, 1 = client→server.
    pub direction: u8,
    /// Alignment padding; always zero.
    pub _pad: u16,
    /// Length of clipboard data following this header.
    pub data_len: u32,
}

/// Supported clipboard MIME types.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClipboardMime {
    /// UTF-8 text (text/plain;charset=utf-8).
    TextPlain = 0,
    /// HTML (text/html).
    TextHtml = 1,
    /// PNG image (image/png).
    ImagePng = 2,
}

impl TryFrom<u8> for ClipboardMime {
    type Error = u8;
    fn try_from(v: u8) -> Result<Self, u8> {
        match v {
            0 => Ok(Self::TextPlain),
            1 => Ok(Self::TextHtml),
            2 => Ok(Self::ImagePng),
            other => Err(other),
        }
    }
}

impl ClipboardHeader {
    /// Size of `ClipboardHeader` in bytes (always 8).
    pub const SIZE: usize = size_of::<Self>();

    /// Create a new clipboard header.
    #[must_use]
    pub fn new(mime: ClipboardMime, direction: u8, data_len: u32) -> Self {
        Self {
            mime_type: mime as u8,
            direction,
            _pad: 0,
            data_len,
        }
    }

    /// Decode the `mime_type` byte into a [`ClipboardMime`] discriminant.
    pub fn mime(&self) -> Result<ClipboardMime, u8> {
        ClipboardMime::try_from(self.mime_type)
    }
}

const _: () = assert!(size_of::<ClipboardHeader>() == 8);

#[cfg(test)]
mod tests {
    use zerocopy::{FromBytes, IntoBytes};

    use super::*;

    #[test]
    fn size_is_8() {
        assert_eq!(ClipboardHeader::SIZE, 8);
    }

    #[test]
    fn round_trip() {
        let h = ClipboardHeader::new(ClipboardMime::TextPlain, 0, 1024);
        let bytes = h.as_bytes();
        let decoded = ClipboardHeader::ref_from_bytes(bytes).unwrap();
        assert_eq!(decoded.mime().unwrap(), ClipboardMime::TextPlain);
        assert_eq!({ decoded.direction }, 0);
        assert_eq!({ decoded.data_len }, 1024);
    }

    #[test]
    fn all_mime_types() {
        for raw in 0u8..=2 {
            assert!(ClipboardMime::try_from(raw).is_ok());
        }
        assert!(ClipboardMime::try_from(3).is_err());
    }
}

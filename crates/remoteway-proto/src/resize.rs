use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

/// Surface resize notification (12 bytes).
///
/// Sent when a surface changes dimensions, either because:
/// - The server detects an output/window resize
/// - The client's local window was resized by the compositor
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct ResizePayload {
    pub surface_id: u16,
    pub _pad: u16,
    pub width: u32,
    pub height: u32,
}

impl ResizePayload {
    pub const SIZE: usize = size_of::<Self>();

    pub fn new(surface_id: u16, width: u32, height: u32) -> Self {
        Self {
            surface_id,
            _pad: 0,
            width,
            height,
        }
    }
}

const _: () = assert!(size_of::<ResizePayload>() == 12);

#[cfg(test)]
mod tests {
    use zerocopy::{FromBytes, IntoBytes};

    use super::*;

    #[test]
    fn size_is_12() {
        assert_eq!(ResizePayload::SIZE, 12);
    }

    #[test]
    fn round_trip() {
        let r = ResizePayload::new(1, 1920, 1080);
        let bytes = r.as_bytes();
        let decoded = ResizePayload::ref_from_bytes(bytes).unwrap();
        assert_eq!({ decoded.surface_id }, 1);
        assert_eq!({ decoded.width }, 1920);
        assert_eq!({ decoded.height }, 1080);
    }
}

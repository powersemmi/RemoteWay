use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

/// Message type discriminant (1 byte).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MsgType {
    FrameUpdate = 0,
    AnchorFrame = 1,
    InputEvent = 2,
    CursorMove = 3,
    Handshake = 4,
    Ack = 5,
    /// Surface resize notification (server→client or client→server).
    Resize = 6,
    /// Clipboard data transfer.
    Clipboard = 7,
    /// Client→server: produce frames at this resolution.
    TargetResolution = 8,
}

impl TryFrom<u8> for MsgType {
    type Error = u8;
    fn try_from(v: u8) -> Result<Self, u8> {
        match v {
            0 => Ok(Self::FrameUpdate),
            1 => Ok(Self::AnchorFrame),
            2 => Ok(Self::InputEvent),
            3 => Ok(Self::CursorMove),
            4 => Ok(Self::Handshake),
            5 => Ok(Self::Ack),
            6 => Ok(Self::Resize),
            7 => Ok(Self::Clipboard),
            8 => Ok(Self::TargetResolution),
            other => Err(other),
        }
    }
}

/// Frame flags bitfield.
pub mod flags {
    pub const COMPRESSED: u8 = 1 << 0;
    pub const LAST_CHUNK: u8 = 1 << 1;
    pub const KEY_FRAME: u8 = 1 << 2;
}

/// Fixed-size 16-byte frame header transmitted before every message payload.
///
/// All fields are little-endian. The struct is `repr(C, packed)` so it can be
/// cast directly to/from bytes via zerocopy without any heap allocation.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct FrameHeader {
    /// 0 = input stream, 1+ = surface id.
    pub stream_id: u16,
    /// Discriminant from [`MsgType`].
    pub msg_type: u8,
    /// Bitfield from [`flags`].
    pub flags: u8,
    pub payload_len: u32,
    pub timestamp_ns: u64,
}

const _: () = assert!(
    std::mem::size_of::<FrameHeader>() == 16,
    "FrameHeader must be exactly 16 bytes"
);

impl FrameHeader {
    pub const SIZE: usize = 16;

    pub fn new(
        stream_id: u16,
        msg_type: MsgType,
        flags: u8,
        payload_len: u32,
        timestamp_ns: u64,
    ) -> Self {
        Self {
            stream_id,
            msg_type: msg_type as u8,
            flags,
            payload_len,
            timestamp_ns,
        }
    }

    pub fn msg_type(&self) -> Result<MsgType, u8> {
        MsgType::try_from(self.msg_type)
    }
}

#[cfg(test)]
mod tests {
    use zerocopy::IntoBytes;

    use super::*;

    #[test]
    fn size_is_16_bytes() {
        assert_eq!(std::mem::size_of::<FrameHeader>(), 16);
    }

    #[test]
    fn round_trip_bytes() {
        let hdr = FrameHeader::new(1, MsgType::FrameUpdate, flags::COMPRESSED, 1024, 999_000);
        let bytes = hdr.as_bytes();
        assert_eq!(bytes.len(), 16);
        let decoded = FrameHeader::ref_from_bytes(bytes).unwrap();
        // Copy packed fields to locals before comparison to avoid unaligned references.
        assert_eq!({ decoded.stream_id }, 1u16.to_le());
        assert_eq!({ decoded.payload_len }, 1024u32.to_le());
        assert_eq!({ decoded.timestamp_ns }, 999_000u64.to_le());
    }

    #[test]
    fn msg_type_round_trip() {
        for raw in 0u8..=8 {
            let mt = MsgType::try_from(raw).unwrap();
            assert_eq!(mt as u8, raw);
        }
        assert!(MsgType::try_from(9u8).is_err());
    }

    #[test]
    fn flags_constants_no_overlap() {
        assert_eq!(flags::COMPRESSED & flags::LAST_CHUNK, 0);
        assert_eq!(flags::COMPRESSED & flags::KEY_FRAME, 0);
        assert_eq!(flags::LAST_CHUNK & flags::KEY_FRAME, 0);
    }

    #[test]
    fn new_all_msg_types() {
        let types = [
            MsgType::FrameUpdate,
            MsgType::AnchorFrame,
            MsgType::InputEvent,
            MsgType::CursorMove,
            MsgType::Handshake,
            MsgType::Ack,
            MsgType::Resize,
            MsgType::Clipboard,
            MsgType::TargetResolution,
        ];
        for mt in types {
            let hdr = FrameHeader::new(0, mt, 0, 0, 0);
            assert_eq!(hdr.msg_type().unwrap(), mt);
        }
    }

    #[test]
    fn size_const_matches_sizeof() {
        assert_eq!(FrameHeader::SIZE, std::mem::size_of::<FrameHeader>());
    }
}

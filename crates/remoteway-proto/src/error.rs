use thiserror::Error;

/// Errors originating from the protocol layer (message decoding,
/// validation, etc.).
#[derive(Debug, Error)]
pub enum ProtoError {
    /// The `msg_type` byte in a [`crate::header::FrameHeader`] does not
    /// correspond to any known [`crate::header::MsgType`] variant.
    #[error("unknown message type: {0:#04x}")]
    UnknownMsgType(u8),
}

impl From<u8> for ProtoError {
    fn from(v: u8) -> Self {
        Self::UnknownMsgType(v)
    }
}

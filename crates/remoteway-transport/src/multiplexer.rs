use std::collections::HashMap;

use remoteway_proto::header::FrameHeader;
use thiserror::Error;
use zerocopy::{FromBytes, IntoBytes};

#[derive(Debug, Error)]
pub enum MultiplexerError {
    #[error("unknown message type: {0}")]
    UnknownMsgType(u8),
    #[error("payload too large: {0} bytes")]
    PayloadTooLarge(u32),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Maximum allowed payload per frame chunk (4 MiB).
pub const MAX_PAYLOAD_LEN: u32 = 4 * 1024 * 1024;

/// A demultiplexed message received from the wire.
#[derive(Debug)]
pub struct IncomingMessage {
    pub header: FrameHeader,
    pub payload: Vec<u8>,
}

/// Serializes a [`FrameHeader`] + payload into `dst`.
pub fn encode_frame(header: &FrameHeader, payload: &[u8], dst: &mut Vec<u8>) {
    dst.extend_from_slice(header.as_bytes());
    dst.extend_from_slice(payload);
}

/// Stateful stream parser that accumulates bytes and emits complete frames.
///
/// Call [`StreamParser::push`] with incoming bytes; it will return any
/// complete [`IncomingMessage`]s found in the buffer.
pub struct StreamParser {
    buf: Vec<u8>,
    /// In-progress reassembly of chunked payloads, keyed by stream_id.
    chunks: HashMap<u16, Vec<u8>>,
}

impl StreamParser {
    pub fn new() -> Self {
        Self {
            buf: Vec::with_capacity(64 * 1024),
            chunks: HashMap::new(),
        }
    }

    /// Feed raw bytes from the wire. Returns any complete messages found.
    pub fn push(&mut self, data: &[u8]) -> Result<Vec<IncomingMessage>, MultiplexerError> {
        self.buf.extend_from_slice(data);
        let mut out = Vec::new();

        loop {
            if self.buf.len() < FrameHeader::SIZE {
                break;
            }
            let hdr = *FrameHeader::ref_from_bytes(&self.buf[..FrameHeader::SIZE])
                .expect("slice is exactly FrameHeader::SIZE bytes");

            let payload_len = { hdr.payload_len } as usize;
            if payload_len > MAX_PAYLOAD_LEN as usize {
                return Err(MultiplexerError::PayloadTooLarge(hdr.payload_len));
            }

            let total = FrameHeader::SIZE + payload_len;
            if self.buf.len() < total {
                break;
            }

            let payload = self.buf[FrameHeader::SIZE..total].to_vec();
            self.buf.drain(..total);

            // Validate msg_type before accepting.
            let _ = hdr.msg_type().map_err(MultiplexerError::UnknownMsgType)?;

            let msg = self.reassemble(hdr, payload)?;
            if let Some(m) = msg {
                out.push(m);
            }
        }

        Ok(out)
    }

    /// Reassemble chunked payloads. Returns `Some` only when the last chunk arrives.
    fn reassemble(
        &mut self,
        hdr: FrameHeader,
        payload: Vec<u8>,
    ) -> Result<Option<IncomingMessage>, MultiplexerError> {
        use remoteway_proto::header::flags;

        let stream_id = { hdr.stream_id };
        let is_last = hdr.flags & flags::LAST_CHUNK != 0;

        if hdr.flags & flags::LAST_CHUNK != 0
            && !self.chunks.contains_key(&stream_id)
            && payload.len() == { hdr.payload_len } as usize
        {
            // Single-chunk message — fast path, no copy.
            return Ok(Some(IncomingMessage {
                header: hdr,
                payload,
            }));
        }

        let buf = self.chunks.entry(stream_id).or_default();
        buf.extend_from_slice(&payload);

        if is_last {
            let full_payload = self.chunks.remove(&stream_id).unwrap_or_default();
            Ok(Some(IncomingMessage {
                header: hdr,
                payload: full_payload,
            }))
        } else {
            Ok(None)
        }
    }

    /// Number of streams currently being reassembled.
    pub fn in_progress_streams(&self) -> usize {
        self.chunks.len()
    }
}

impl Default for StreamParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use remoteway_proto::header::{FrameHeader, MsgType, flags};

    use super::*;

    fn make_frame(stream_id: u16, msg_type: MsgType, flags: u8, payload: &[u8]) -> Vec<u8> {
        let hdr = FrameHeader::new(stream_id, msg_type, flags, payload.len() as u32, 0);
        let mut out = Vec::new();
        encode_frame(&hdr, payload, &mut out);
        out
    }

    #[test]
    fn single_chunk_message() {
        let mut p = StreamParser::new();
        let data = make_frame(1, MsgType::FrameUpdate, flags::LAST_CHUNK, b"hello");
        let msgs = p.push(&data).unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].payload, b"hello");
    }

    #[test]
    fn two_messages_in_one_push() {
        let mut p = StreamParser::new();
        let mut data = make_frame(1, MsgType::FrameUpdate, flags::LAST_CHUNK, b"first");
        data.extend(make_frame(
            2,
            MsgType::AnchorFrame,
            flags::LAST_CHUNK,
            b"second",
        ));
        let msgs = p.push(&data).unwrap();
        assert_eq!(msgs.len(), 2);
    }

    #[test]
    fn partial_header_waits() {
        let mut p = StreamParser::new();
        let data = make_frame(1, MsgType::FrameUpdate, flags::LAST_CHUNK, b"x");
        // Send only 8 of the 16-byte header.
        let msgs = p.push(&data[..8]).unwrap();
        assert!(msgs.is_empty());
        let msgs = p.push(&data[8..]).unwrap();
        assert_eq!(msgs.len(), 1);
    }

    #[test]
    fn partial_payload_waits() {
        let mut p = StreamParser::new();
        let data = make_frame(1, MsgType::FrameUpdate, flags::LAST_CHUNK, b"abcdef");
        let mid = data.len() / 2;
        let msgs = p.push(&data[..mid]).unwrap();
        assert!(msgs.is_empty());
        let msgs = p.push(&data[mid..]).unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].payload, b"abcdef");
    }

    #[test]
    fn multi_chunk_reassembly() {
        let mut p = StreamParser::new();
        // Chunk 1: no LAST_CHUNK flag.
        let chunk1 = make_frame(3, MsgType::FrameUpdate, 0, b"part1-");
        // Chunk 2: LAST_CHUNK flag.
        let chunk2 = make_frame(3, MsgType::FrameUpdate, flags::LAST_CHUNK, b"part2");
        let mut msgs = p.push(&chunk1).unwrap();
        assert!(msgs.is_empty());
        msgs = p.push(&chunk2).unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].payload, b"part1-part2");
    }

    #[test]
    fn interleaved_streams() {
        let mut p = StreamParser::new();
        let a1 = make_frame(1, MsgType::FrameUpdate, 0, b"a1-");
        let b1 = make_frame(2, MsgType::FrameUpdate, 0, b"b1-");
        let a2 = make_frame(1, MsgType::FrameUpdate, flags::LAST_CHUNK, b"a2");
        let b2 = make_frame(2, MsgType::FrameUpdate, flags::LAST_CHUNK, b"b2");

        let mut all = a1;
        all.extend(b1);
        all.extend(a2);
        all.extend(b2);

        let msgs = p.push(&all).unwrap();
        assert_eq!(msgs.len(), 2);
        let payloads: Vec<_> = msgs.iter().map(|m| m.payload.clone()).collect();
        assert!(payloads.contains(&b"a1-a2".to_vec()));
        assert!(payloads.contains(&b"b1-b2".to_vec()));
    }

    #[test]
    fn oversized_payload_returns_error() {
        let mut p = StreamParser::new();
        let hdr = FrameHeader::new(
            1,
            MsgType::FrameUpdate,
            flags::LAST_CHUNK,
            MAX_PAYLOAD_LEN + 1,
            0,
        );
        let mut data = Vec::new();
        encode_frame(&hdr, &[], &mut data);
        assert!(p.push(&data).is_err());
    }

    #[test]
    fn unknown_msg_type_returns_error() {
        let mut p = StreamParser::new();
        // Craft a header with invalid msg_type byte.
        let mut hdr = FrameHeader::new(1, MsgType::FrameUpdate, flags::LAST_CHUNK, 0, 0);
        hdr.msg_type = 0xFF;
        let mut data = Vec::new();
        encode_frame(&hdr, &[], &mut data);
        assert!(p.push(&data).is_err());
    }

    #[test]
    fn many_concurrent_streams_32() {
        let mut p = StreamParser::new();
        let mut wire = Vec::new();
        // 32 streams, each with 2 chunks.
        for sid in 0..32u16 {
            let chunk1 = format!("s{sid}-1-");
            wire.extend(make_frame(sid, MsgType::FrameUpdate, 0, chunk1.as_bytes()));
        }
        for sid in 0..32u16 {
            let chunk2 = format!("s{sid}-2");
            wire.extend(make_frame(
                sid,
                MsgType::FrameUpdate,
                flags::LAST_CHUNK,
                chunk2.as_bytes(),
            ));
        }
        let msgs = p.push(&wire).unwrap();
        assert_eq!(msgs.len(), 32);
        for sid in 0..32u16 {
            let expected = format!("s{sid}-1-s{sid}-2");
            assert!(
                msgs.iter().any(|m| m.payload == expected.as_bytes()),
                "missing reassembled message for stream {sid}"
            );
        }
    }

    #[test]
    fn byte_by_byte_push() {
        let mut p = StreamParser::new();
        let data = make_frame(1, MsgType::FrameUpdate, flags::LAST_CHUNK, b"payload");
        let mut total_msgs = Vec::new();
        for &b in &data {
            let msgs = p.push(&[b]).unwrap();
            total_msgs.extend(msgs);
        }
        assert_eq!(total_msgs.len(), 1);
        assert_eq!(total_msgs[0].payload, b"payload");
    }

    #[test]
    fn in_progress_streams_tracking() {
        let mut p = StreamParser::new();
        assert_eq!(p.in_progress_streams(), 0);
        // Non-last chunk for stream 1.
        p.push(&make_frame(1, MsgType::FrameUpdate, 0, b"a"))
            .unwrap();
        assert_eq!(p.in_progress_streams(), 1);
        // Non-last chunk for stream 2.
        p.push(&make_frame(2, MsgType::FrameUpdate, 0, b"b"))
            .unwrap();
        assert_eq!(p.in_progress_streams(), 2);
        // Last chunk for stream 1.
        p.push(&make_frame(
            1,
            MsgType::FrameUpdate,
            flags::LAST_CHUNK,
            b"c",
        ))
        .unwrap();
        assert_eq!(p.in_progress_streams(), 1);
        // Last chunk for stream 2.
        p.push(&make_frame(
            2,
            MsgType::FrameUpdate,
            flags::LAST_CHUNK,
            b"d",
        ))
        .unwrap();
        assert_eq!(p.in_progress_streams(), 0);
    }

    #[test]
    fn default_trait() {
        let p = StreamParser::default();
        assert_eq!(p.in_progress_streams(), 0);
    }

    #[test]
    fn encode_frame_output_length() {
        let payload = b"test";
        let hdr = FrameHeader::new(0, MsgType::Ack, flags::LAST_CHUNK, payload.len() as u32, 0);
        let mut dst = Vec::new();
        encode_frame(&hdr, payload, &mut dst);
        assert_eq!(dst.len(), FrameHeader::SIZE + payload.len());
    }
}

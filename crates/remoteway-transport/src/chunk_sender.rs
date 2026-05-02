//! Splits payloads into fixed-size chunks for transmission over the multiplexed protocol.

use remoteway_proto::header::{FrameHeader, MsgType, flags};
use zerocopy::IntoBytes;

/// Splits a large payload into chunks and writes them into `dst`.
///
/// Each chunk gets a `FrameHeader` with `LAST_CHUNK` set only on the final one.
/// `chunk_size` is the maximum payload bytes per chunk.
pub fn split_into_chunks(
    stream_id: u16,
    msg_type: MsgType,
    extra_flags: u8,
    payload: &[u8],
    timestamp_ns: u64,
    chunk_size: usize,
    dst: &mut Vec<u8>,
) {
    debug_assert!(chunk_size > 0, "chunk_size must be > 0");

    if payload.is_empty() {
        let hdr = FrameHeader::new(
            stream_id,
            msg_type,
            extra_flags | flags::LAST_CHUNK,
            0,
            timestamp_ns,
        );
        dst.extend_from_slice(hdr.as_bytes());
        return;
    }

    let mut offset = 0;
    while offset < payload.len() {
        let end = (offset + chunk_size).min(payload.len());
        let chunk = &payload[offset..end];
        let is_last = end == payload.len();
        let chunk_flags = if is_last {
            extra_flags | flags::LAST_CHUNK
        } else {
            extra_flags
        };
        let hdr = FrameHeader::new(
            stream_id,
            msg_type,
            chunk_flags,
            chunk.len() as u32,
            timestamp_ns,
        );
        dst.extend_from_slice(hdr.as_bytes());
        dst.extend_from_slice(chunk);
        offset = end;
    }
}

/// Number of chunks that would be produced for a given payload length.
#[must_use]
pub fn chunk_count(payload_len: usize, chunk_size: usize) -> usize {
    if payload_len == 0 || chunk_size == 0 {
        return 1;
    }
    payload_len.div_ceil(chunk_size)
}

#[cfg(test)]
mod tests {
    use remoteway_proto::header::{FrameHeader, MsgType, flags};
    use zerocopy::FromBytes;

    use super::*;

    fn decode_frames(mut data: &[u8]) -> Vec<(FrameHeader, Vec<u8>)> {
        let mut out = Vec::new();
        while data.len() >= FrameHeader::SIZE {
            let hdr = *FrameHeader::ref_from_bytes(&data[..FrameHeader::SIZE]).unwrap();
            let plen = { hdr.payload_len } as usize;
            let payload = data[FrameHeader::SIZE..FrameHeader::SIZE + plen].to_vec();
            data = &data[FrameHeader::SIZE + plen..];
            out.push((hdr, payload));
        }
        out
    }

    #[test]
    fn single_chunk_small_payload() {
        let mut dst = Vec::new();
        split_into_chunks(1, MsgType::FrameUpdate, 0, b"hello", 42, 1024, &mut dst);
        let frames = decode_frames(&dst);
        assert_eq!(frames.len(), 1);
        assert_ne!(frames[0].0.flags & flags::LAST_CHUNK, 0);
        assert_eq!(frames[0].1, b"hello");
    }

    #[test]
    fn multi_chunk_split() {
        let payload: Vec<u8> = (0..10).collect();
        let mut dst = Vec::new();
        split_into_chunks(1, MsgType::FrameUpdate, 0, &payload, 0, 3, &mut dst);
        let frames = decode_frames(&dst);
        // 10 bytes / 3 = 4 chunks (3+3+3+1)
        assert_eq!(frames.len(), 4);
        assert_eq!(frames[3].0.flags & flags::LAST_CHUNK, flags::LAST_CHUNK);
        for f in &frames[..3] {
            assert_eq!(f.0.flags & flags::LAST_CHUNK, 0);
        }
        // Reassemble and verify.
        let reassembled: Vec<u8> = frames.iter().flat_map(|f| f.1.clone()).collect();
        assert_eq!(reassembled, payload);
    }

    #[test]
    fn empty_payload_produces_one_last_chunk_frame() {
        let mut dst = Vec::new();
        split_into_chunks(1, MsgType::AnchorFrame, 0, &[], 0, 512, &mut dst);
        let frames = decode_frames(&dst);
        assert_eq!(frames.len(), 1);
        assert_ne!(frames[0].0.flags & flags::LAST_CHUNK, 0);
        assert!(frames[0].1.is_empty());
    }

    #[test]
    fn exact_chunk_boundary() {
        let payload = vec![0u8; 6];
        let mut dst = Vec::new();
        split_into_chunks(1, MsgType::FrameUpdate, 0, &payload, 0, 3, &mut dst);
        let frames = decode_frames(&dst);
        assert_eq!(frames.len(), 2);
    }

    #[test]
    fn extra_flags_preserved_on_last_chunk() {
        let mut dst = Vec::new();
        split_into_chunks(
            1,
            MsgType::FrameUpdate,
            flags::COMPRESSED,
            b"data",
            0,
            1024,
            &mut dst,
        );
        let frames = decode_frames(&dst);
        assert_eq!(frames.len(), 1);
        assert_ne!(frames[0].0.flags & flags::COMPRESSED, 0);
    }

    #[test]
    fn chunk_count_calculation() {
        assert_eq!(chunk_count(0, 512), 1);
        assert_eq!(chunk_count(512, 512), 1);
        assert_eq!(chunk_count(513, 512), 2);
        assert_eq!(chunk_count(1024, 512), 2);
        assert_eq!(chunk_count(1025, 512), 3);
    }

    #[test]
    fn chunk_size_one() {
        let payload = b"hello";
        let mut dst = Vec::new();
        split_into_chunks(1, MsgType::FrameUpdate, 0, payload, 0, 1, &mut dst);
        let frames = decode_frames(&dst);
        assert_eq!(frames.len(), 5);
        for f in &frames[..4] {
            assert_eq!(f.0.flags & flags::LAST_CHUNK, 0);
            assert_eq!(f.1.len(), 1);
        }
        assert_ne!(frames[4].0.flags & flags::LAST_CHUNK, 0);
        let reassembled: Vec<u8> = frames.iter().flat_map(|f| f.1.clone()).collect();
        assert_eq!(reassembled, payload);
    }

    #[test]
    fn timestamp_preserved_all_chunks() {
        let payload = vec![0u8; 100];
        let ts = 1_234_567_890u64;
        let mut dst = Vec::new();
        split_into_chunks(5, MsgType::FrameUpdate, 0, &payload, ts, 30, &mut dst);
        let frames = decode_frames(&dst);
        assert!(frames.len() > 1);
        for f in &frames {
            assert_eq!({ f.0.timestamp_ns }, ts);
        }
    }

    #[test]
    fn stream_id_preserved_all_chunks() {
        let payload = vec![0u8; 50];
        let mut dst = Vec::new();
        split_into_chunks(42, MsgType::FrameUpdate, 0, &payload, 0, 20, &mut dst);
        let frames = decode_frames(&dst);
        for f in &frames {
            assert_eq!({ f.0.stream_id }, 42);
        }
    }
}

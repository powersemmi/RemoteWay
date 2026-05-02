use proptest::prelude::*;
use remoteway_proto::header::{FrameHeader, MsgType, flags};
use remoteway_transport::multiplexer::{StreamParser, encode_frame};

// ── helpers ──────────────────────────────────────────────────────────────────

fn arb_stream_id() -> impl Strategy<Value = u16> {
    0u16..=63u16 // keep small so reassembly map stays manageable
}

fn arb_msg_type_byte() -> impl Strategy<Value = u8> {
    0u8..=5u8
}

fn arb_payload(max: usize) -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(any::<u8>(), 0..max)
}

/// Build a single well-formed frame (`LAST_CHUNK` set) as raw bytes.
fn frame_bytes(stream_id: u16, msg_type: u8, payload: &[u8]) -> Vec<u8> {
    let mut hdr = FrameHeader::new(
        stream_id,
        MsgType::FrameUpdate,
        flags::LAST_CHUNK,
        payload.len() as u32,
        0,
    );
    hdr.msg_type = msg_type;
    let mut out = Vec::new();
    encode_frame(&hdr, payload, &mut out);
    out
}

// ── proptests ────────────────────────────────────────────────────────────────

proptest! {
    /// Single frame with arbitrary payload round-trips through StreamParser without panic.
    #[test]
    fn single_frame_no_loss(
        stream_id in arb_stream_id(),
        msg_type in arb_msg_type_byte(),
        payload in arb_payload(4096),
    ) {
        let raw = frame_bytes(stream_id, msg_type, &payload);
        let mut parser = StreamParser::new();
        let msgs = parser.push(&raw).unwrap();
        prop_assert_eq!(msgs.len(), 1);
        prop_assert_eq!(&msgs[0].payload, &payload);
    }

    /// Feeding bytes one at a time produces the same result as feeding all at once.
    #[test]
    fn byte_by_byte_same_result(
        stream_id in arb_stream_id(),
        msg_type in arb_msg_type_byte(),
        payload in arb_payload(256),
    ) {
        let raw = frame_bytes(stream_id, msg_type, &payload);

        // All at once.
        let mut p1 = StreamParser::new();
        let bulk = p1.push(&raw).unwrap();

        // Byte by byte.
        let mut p2 = StreamParser::new();
        let mut incremental: Vec<_> = Vec::new();
        for byte in &raw {
            incremental.extend(p2.push(&[*byte]).unwrap());
        }

        prop_assert_eq!(bulk.len(), incremental.len());
        for (a, b) in bulk.iter().zip(incremental.iter()) {
            prop_assert_eq!(&a.payload, &b.payload);
        }
    }

    /// N concatenated frames produce exactly N messages, in order, without duplication.
    #[test]
    fn n_frames_no_duplication(
        payloads in prop::collection::vec(arb_payload(128), 1..=16),
    ) {
        let mut raw = Vec::new();
        for (i, p) in payloads.iter().enumerate() {
            raw.extend(frame_bytes(i as u16, 0, p));
        }

        let mut parser = StreamParser::new();
        let msgs = parser.push(&raw).unwrap();
        prop_assert_eq!(msgs.len(), payloads.len());
        for (msg, expected) in msgs.iter().zip(payloads.iter()) {
            prop_assert_eq!(&msg.payload, expected);
        }
    }

    /// Random split point: feeding frame in two halves produces the same single message.
    #[test]
    fn split_at_arbitrary_offset(
        stream_id in arb_stream_id(),
        msg_type in arb_msg_type_byte(),
        payload in arb_payload(512),
        split in 0usize..=511usize,
    ) {
        let raw = frame_bytes(stream_id, msg_type, &payload);
        let split = split.min(raw.len());

        let mut parser = StreamParser::new();
        let mut msgs = parser.push(&raw[..split]).unwrap();
        msgs.extend(parser.push(&raw[split..]).unwrap());

        prop_assert_eq!(msgs.len(), 1);
        prop_assert_eq!(&msgs[0].payload, &payload);
    }

    /// Frames for different stream_ids interleaved: each stream's payload is preserved.
    ///
    /// Strategy: stream 10 sends two chunks (no LAST_CHUNK, then LAST_CHUNK),
    /// with stream 20's single frame inserted between them. The parser must
    /// reassemble each stream independently without mixing payloads.
    #[test]
    fn interleaved_streams_no_mix(
        part_a in arb_payload(64),
        part_b in arb_payload(64),
        p2 in arb_payload(128),
    ) {
        // Stream 10 chunk 1: no LAST_CHUNK.
        let hdr1a = FrameHeader::new(
            10,
            MsgType::FrameUpdate,
            0,
            part_a.len() as u32,
            0,
        );
        // Stream 10 chunk 2: LAST_CHUNK.
        let hdr1b = FrameHeader::new(
            10,
            MsgType::FrameUpdate,
            flags::LAST_CHUNK,
            part_b.len() as u32,
            0,
        );
        // Stream 20: single LAST_CHUNK frame.
        let f2 = frame_bytes(20, 0, &p2);

        let mut raw = Vec::new();
        encode_frame(&hdr1a, &part_a, &mut raw);
        raw.extend(&f2);
        encode_frame(&hdr1b, &part_b, &mut raw);

        let mut parser = StreamParser::new();
        let msgs = parser.push(&raw).unwrap();

        // Exactly 2 messages: stream-20 (single chunk, emitted first) and stream-10 (reassembled).
        prop_assert_eq!(msgs.len(), 2);

        prop_assert_eq!({ msgs[0].header.stream_id }, 20u16);
        prop_assert_eq!(&msgs[0].payload, &p2);

        prop_assert_eq!({ msgs[1].header.stream_id }, 10u16);
        let expected_10: Vec<u8> = part_a.iter().chain(part_b.iter()).copied().collect();
        prop_assert_eq!(&msgs[1].payload, &expected_10);
    }
}

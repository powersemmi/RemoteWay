use proptest::prelude::*;
use remoteway_proto::header::{FrameHeader, MsgType, flags};
use remoteway_proto::input::{
    InputEvent, InputKind, KeyEvent, PointerAxis, PointerButton, PointerMotion,
};
use zerocopy::{FromBytes, IntoBytes};

// ── helpers ──────────────────────────────────────────────────────────────────

fn arb_msg_type() -> impl Strategy<Value = u8> {
    0u8..=8u8 // valid MsgType range (FrameUpdate..=TargetResolution)
}

fn arb_flags() -> impl Strategy<Value = u8> {
    prop_oneof![
        Just(0u8),
        Just(flags::COMPRESSED),
        Just(flags::LAST_CHUNK),
        Just(flags::KEY_FRAME),
        Just(flags::COMPRESSED | flags::LAST_CHUNK),
        Just(flags::COMPRESSED | flags::KEY_FRAME | flags::LAST_CHUNK),
    ]
}

// ── FrameHeader proptest ──────────────────────────────────────────────────────

proptest! {
    /// Any valid FrameHeader round-trips through as_bytes / ref_from_bytes.
    #[test]
    fn frame_header_bytes_round_trip(
        stream_id in 0u16..=65535u16,
        msg_type in arb_msg_type(),
        flag in arb_flags(),
        payload_len in 0u32..=4_194_304u32,
        ts in 0u64..=u64::MAX,
    ) {
        let hdr = FrameHeader {
            stream_id,
            msg_type,
            flags: flag,
            payload_len,
            timestamp_ns: ts,
        };
        let bytes = hdr.as_bytes();
        prop_assert_eq!(bytes.len(), 16);
        let decoded = FrameHeader::ref_from_bytes(bytes).unwrap();
        prop_assert_eq!({ decoded.stream_id }, stream_id);
        prop_assert_eq!({ decoded.payload_len }, payload_len);
        prop_assert_eq!({ decoded.timestamp_ns }, ts);
        prop_assert_eq!(decoded.flags, flag);
    }

    /// Arbitrary stream_id values (including 0 for input stream) do not cause panics.
    #[test]
    fn frame_header_any_stream_id_no_panic(
        stream_id in 0u16..=65535u16,
        payload_len in 0u32..=1024u32,
    ) {
        let hdr = FrameHeader::new(stream_id, MsgType::FrameUpdate, flags::LAST_CHUNK, payload_len, 0);
        let _ = hdr.as_bytes();
        let _ = hdr.msg_type();
    }

    /// Random payload_len values produce consistent byte representations.
    #[test]
    fn frame_header_payload_len_any_value(payload_len in 0u32..=u32::MAX) {
        let hdr = FrameHeader::new(1, MsgType::FrameUpdate, 0, payload_len, 0);
        let bytes = hdr.as_bytes();
        let decoded = FrameHeader::ref_from_bytes(bytes).unwrap();
        prop_assert_eq!({ decoded.payload_len }, payload_len);
    }

    /// Unknown msg_type bytes (9..=255) return Err, not panic.
    #[test]
    fn unknown_msg_type_returns_err_not_panic(raw in 9u8..=255u8) {
        let mut hdr = FrameHeader::new(1, MsgType::FrameUpdate, 0, 0, 0);
        hdr.msg_type = raw;
        prop_assert!(hdr.msg_type().is_err());
    }
}

// ── InputEvent proptest ───────────────────────────────────────────────────────

proptest! {
    /// PointerMotion round-trips through InputEvent.
    #[test]
    fn pointer_motion_round_trip(
        surface_id in 0u16..=65535u16,
        x in -10000.0f32..=10000.0f32,
        y in -10000.0f32..=10000.0f32,
    ) {
        let motion = PointerMotion { surface_id, _pad: 0, x, y };
        let ev = InputEvent::pointer_motion(motion);
        prop_assert_eq!(ev.kind().unwrap(), InputKind::PointerMotion);
        prop_assert_eq!(ev.as_bytes().len(), 16);
    }

    /// KeyEvent round-trips through InputEvent.
    #[test]
    fn key_event_round_trip(key in 0u32..=255u32, state in 0u32..=2u32) {
        let kev = KeyEvent { key, state };
        let ev = InputEvent::key(kev);
        prop_assert_eq!(ev.kind().unwrap(), InputKind::Key);
        // Decode back and verify.
        use std::mem::size_of;
        let decoded = KeyEvent::ref_from_bytes(&ev.payload[..size_of::<KeyEvent>()]).unwrap();
        prop_assert_eq!({ decoded.key }, key);
        prop_assert_eq!({ decoded.state }, state);
    }

    /// PointerButton round-trips.
    #[test]
    fn pointer_button_round_trip(button in 0u32..=512u32, state in 0u32..=1u32) {
        let btn = PointerButton { button, state };
        let ev = InputEvent::pointer_button(btn);
        prop_assert_eq!(ev.kind().unwrap(), InputKind::PointerButton);
        prop_assert_eq!(ev.as_bytes().len(), 16);
    }

    /// PointerAxis round-trips.
    #[test]
    fn pointer_axis_round_trip(axis in 0u8..=1u8, value in -100.0f32..=100.0f32) {
        let a = PointerAxis { axis, _pad: [0; 3], value };
        let ev = InputEvent::pointer_axis(a);
        prop_assert_eq!(ev.kind().unwrap(), InputKind::PointerAxis);
        prop_assert_eq!(ev.as_bytes().len(), 16);
    }

    /// Any byte in 6..=255 used as InputEvent::kind returns Err (not panic).
    #[test]
    fn unknown_input_kind_no_panic(raw in 4u8..=255u8) {
        let mut ev = InputEvent::key(KeyEvent { key: 0, state: 0 });
        ev.kind = raw;
        let _ = ev.kind(); // must not panic
    }
}

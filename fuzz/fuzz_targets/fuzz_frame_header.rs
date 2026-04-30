#![no_main]

use libfuzzer_sys::fuzz_target;
use remoteway_proto::header::FrameHeader;
use zerocopy::FromBytes;

fuzz_target!(|data: &[u8]| {
    // Attempt to parse a FrameHeader from arbitrary bytes.
    // Must never panic regardless of input.
    if data.len() >= FrameHeader::SIZE {
        if let Ok(hdr) = FrameHeader::ref_from_bytes(&data[..FrameHeader::SIZE]) {
            // Access packed fields (forces copy, must not panic).
            let _ = hdr.msg_type();
            let _ = hdr.stream_id;
            let _ = hdr.flags;
            let _ = hdr.payload_len;
            let _ = hdr.timestamp_ns;
        }
    }
});

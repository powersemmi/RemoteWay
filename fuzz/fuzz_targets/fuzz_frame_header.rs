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
            hdr.msg_type();
            hdr.stream_id;
            hdr.flags;
            hdr.payload_len;
            hdr.timestamp_ns;
        }
    }
});

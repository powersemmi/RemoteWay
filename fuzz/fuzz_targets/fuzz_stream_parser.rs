#![no_main]

use libfuzzer_sys::fuzz_target;
use remoteway_transport::multiplexer::StreamParser;

fuzz_target!(|data: &[u8]| {
    // Feed arbitrary bytes into the StreamParser.
    // Must never panic, regardless of malformed input.
    let mut parser = StreamParser::new();

    // Feed all at once.
    let _ = parser.push(data);

    // Feed byte-by-byte (different state machine paths).
    let mut parser2 = StreamParser::new();
    for byte in data {
        let _ = parser2.push(std::slice::from_ref(byte));
    }
});

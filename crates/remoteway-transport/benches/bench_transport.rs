use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use remoteway_proto::header::{FrameHeader, MsgType, flags};
use remoteway_transport::multiplexer::{StreamParser, encode_frame};

fn make_frames(count: usize, payload_size: usize) -> Vec<u8> {
    let payload = vec![0xABu8; payload_size];
    let mut out = Vec::with_capacity((FrameHeader::SIZE + payload_size) * count);
    for i in 0..count {
        let hdr = FrameHeader::new(
            (i % 8) as u16,
            MsgType::FrameUpdate,
            flags::LAST_CHUNK,
            payload_size as u32,
            i as u64,
        );
        encode_frame(&hdr, &payload, &mut out);
    }
    out
}

/// Throughput of `StreamParser::push` for different payload sizes.
fn bench_parser_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("StreamParser/throughput");

    for payload_size in [64usize, 1024, 16 * 1024, 64 * 1024] {
        let count = (4 * 1024 * 1024) / payload_size.max(1);
        let data = make_frames(count, payload_size);
        let total_bytes = data.len() as u64;

        group.throughput(Throughput::Bytes(total_bytes));
        group.bench_with_input(
            BenchmarkId::new("payload_bytes", payload_size),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut parser = StreamParser::new();
                    let msgs = parser.push(data).unwrap();
                    assert_eq!(msgs.len(), count);
                });
            },
        );
    }

    group.finish();
}

/// Latency of assembling a single chunk message.
fn bench_single_frame_latency(c: &mut Criterion) {
    let payload = vec![0u8; 1024];
    let hdr = FrameHeader::new(1, MsgType::FrameUpdate, flags::LAST_CHUNK, 1024, 0);
    let mut frame = Vec::new();
    encode_frame(&hdr, &payload, &mut frame);

    c.bench_function("StreamParser/single_frame_1KiB", |b| {
        b.iter(|| {
            let mut parser = StreamParser::new();
            let msgs = parser.push(&frame).unwrap();
            assert_eq!(msgs.len(), 1);
        });
    });
}

/// Throughput of `StreamParser` fed byte by byte (worst case fragmentation).
fn bench_byte_by_byte(c: &mut Criterion) {
    // 16 small frames fed one byte at a time.
    let data = make_frames(16, 32);

    c.bench_function("StreamParser/byte_by_byte_16_frames", |b| {
        b.iter(|| {
            let mut parser = StreamParser::new();
            let mut total = 0usize;
            for byte in &data {
                total += parser.push(&[*byte]).unwrap().len();
            }
            assert_eq!(total, 16);
        });
    });
}

criterion_group!(
    benches,
    bench_parser_throughput,
    bench_single_frame_latency,
    bench_byte_by_byte,
);
criterion_main!(benches);

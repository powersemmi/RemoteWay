//! Stress test: synthetic 4K full-damage frames through compress→serialize→deserialize→decompress.
//!
//! Verifies the entire wire pipeline round-trip under sustained load.

use remoteway_compress::delta::DamageRect;
use remoteway_compress::pipeline::{compress_frame, decompress_frame};
use remoteway_proto::frame::{FrameMeta, WireRegion};
use zerocopy::{FromBytes, IntoBytes};

const WIDTH: u32 = 3840;
const HEIGHT: u32 = 2160;
const STRIDE: u32 = WIDTH * 4;
const FRAME_SIZE: usize = (STRIDE * HEIGHT) as usize;
const NUM_FRAMES: usize = 30;

fn generate_frame(seed: u8) -> Vec<u8> {
    (0..FRAME_SIZE)
        .map(|i| ((i as u64).wrapping_mul(seed as u64 + 7).wrapping_add(3)) as u8)
        .collect()
}

fn serialize_payload(
    width: u32,
    height: u32,
    stride: u32,
    regions: &[DamageRect],
    compressed: &remoteway_compress::pipeline::CompressedFrame,
) -> Vec<u8> {
    let meta = FrameMeta::new(width, height, stride, regions.len() as u32);
    let mut payload = Vec::new();
    payload.extend_from_slice(meta.as_bytes());
    for (i, rect) in regions.iter().enumerate() {
        let wr = WireRegion::new(
            rect.x,
            rect.y,
            rect.width,
            rect.height,
            compressed.region_sizes[i] as u32,
        );
        payload.extend_from_slice(wr.as_bytes());
    }
    payload.extend_from_slice(&compressed.data);
    payload
}

fn deserialize_payload(
    payload: &[u8],
) -> (
    u32,
    u32,
    u32,
    Vec<DamageRect>,
    remoteway_compress::pipeline::CompressedFrame,
) {
    let meta = FrameMeta::ref_from_bytes(&payload[..FrameMeta::SIZE]).unwrap();
    let width = meta.width;
    let height = meta.height;
    let stride = meta.stride;
    let num = meta.num_regions as usize;

    let mut regions = Vec::with_capacity(num);
    let mut offsets = Vec::with_capacity(num);
    let mut sizes = Vec::with_capacity(num);
    let mut data_off = 0usize;

    for i in 0..num {
        let off = FrameMeta::SIZE + i * WireRegion::SIZE;
        let wr = WireRegion::ref_from_bytes(&payload[off..off + WireRegion::SIZE]).unwrap();
        regions.push(DamageRect::new(wr.x, wr.y, wr.w, wr.h));
        offsets.push(data_off);
        let cs = wr.compressed_size as usize;
        sizes.push(cs);
        data_off += cs;
    }

    let data_start = FrameMeta::SIZE + num * WireRegion::SIZE;
    let data = payload[data_start..].to_vec();

    (
        width,
        height,
        stride,
        regions,
        remoteway_compress::pipeline::CompressedFrame {
            data,
            region_offsets: offsets,
            region_sizes: sizes,
            stats: Default::default(),
        },
    )
}

#[test]
fn stress_4k_full_damage_30_frames() {
    let mut previous = vec![0u8; FRAME_SIZE];
    let regions = vec![DamageRect::new(0, 0, WIDTH, HEIGHT)];

    for i in 0..NUM_FRAMES {
        let current = generate_frame(i as u8);

        // Compress.
        let compressed = compress_frame(&current, &previous, STRIDE as usize, &regions);

        // Serialize to wire format.
        let payload = serialize_payload(WIDTH, HEIGHT, STRIDE, &regions, &compressed);

        // Deserialize from wire format.
        let (dw, dh, ds, dregs, dc) = deserialize_payload(&payload);
        assert_eq!(dw, WIDTH);
        assert_eq!(dh, HEIGHT);
        assert_eq!(ds, STRIDE);
        assert_eq!(dregs.len(), 1);

        // Decompress.
        let mut output = Vec::new();
        decompress_frame(&dc, &previous, ds as usize, &dregs, &mut output).unwrap();
        assert_eq!(output, current, "frame {i} mismatch after round-trip");

        previous = current;
    }
}

#[test]
fn stress_4k_partial_damage_100_frames() {
    let mut previous = vec![0u8; FRAME_SIZE];
    // 5% damage: one region covering ~5% of the frame.
    let damage_w = WIDTH / 5;
    let damage_h = HEIGHT / 4;
    let regions = vec![DamageRect::new(100, 100, damage_w, damage_h)];

    for i in 0..100 {
        let current = generate_frame(i as u8);

        let compressed = compress_frame(&current, &previous, STRIDE as usize, &regions);
        let payload = serialize_payload(WIDTH, HEIGHT, STRIDE, &regions, &compressed);
        let (_, _, ds, dregs, dc) = deserialize_payload(&payload);

        let mut output = Vec::new();
        decompress_frame(&dc, &previous, ds as usize, &dregs, &mut output).unwrap();

        // Verify damaged pixels match.
        for rect in &dregs {
            for row in 0..rect.height as usize {
                let y = rect.y as usize + row;
                let x_start = rect.x as usize * 4;
                let x_end = x_start + rect.width as usize * 4;
                let off = y * ds as usize;
                assert_eq!(
                    &output[off + x_start..off + x_end],
                    &current[off + x_start..off + x_end],
                    "frame {i} damaged region mismatch"
                );
            }
        }

        previous = output;
    }
}

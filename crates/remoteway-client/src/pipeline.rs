use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use remoteway_compress::delta::DamageRect;
use remoteway_compress::pipeline::{CompressedFrame, decompress_frame};
use remoteway_display::{DisplayFrame, DisplayThread};
use remoteway_interpolate::{GpuFrame, InterpolationManager};
use remoteway_proto::frame::{FrameMeta, WireRegion};
use remoteway_proto::header::{FrameHeader, MsgType, flags};
use remoteway_proto::input::InputEvent;
use remoteway_proto::resize::ResizePayload;
use remoteway_transport::ssh_transport::TransportSender;
use tracing::{debug, info, warn};
use zerocopy::{FromBytes, IntoBytes};

/// Build the client handshake payload.
pub fn build_handshake() -> Vec<u8> {
    use remoteway_proto::handshake::{HandshakePayload, capture_flags, compress_flags};

    let cap = capture_flags::WLR_SCREENCOPY | capture_flags::EXT_IMAGE_CAPTURE;
    let comp = compress_flags::LZ4;

    let hs = HandshakePayload::new(cap, comp);
    let hdr = FrameHeader::new(0, MsgType::Handshake, flags::LAST_CHUNK, 8, 0);

    let mut wire = Vec::with_capacity(24);
    wire.extend_from_slice(hdr.as_bytes());
    wire.extend_from_slice(hs.as_bytes());
    wire
}

/// Build a `TargetResolution` message to tell the server to downscale to `width×height`.
pub fn build_target_resolution(width: u32, height: u32) -> Vec<u8> {
    use remoteway_proto::target_resolution::TargetResolutionPayload;

    let payload = TargetResolutionPayload::new(width, height);
    let hdr = FrameHeader::new(
        0,
        MsgType::TargetResolution,
        flags::LAST_CHUNK,
        TargetResolutionPayload::SIZE as u32,
        0,
    );

    let mut wire = Vec::with_capacity(FrameHeader::SIZE + TargetResolutionPayload::SIZE);
    wire.extend_from_slice(hdr.as_bytes());
    wire.extend_from_slice(payload.as_bytes());
    wire
}

/// Deserialize a frame payload from wire bytes using zerocopy.
///
/// Returns (width, height, stride, `damage_regions`, `CompressedFrame`).
pub fn deserialize_frame_payload(
    payload: &[u8],
) -> Result<(u32, u32, u32, Vec<DamageRect>, CompressedFrame)> {
    anyhow::ensure!(
        payload.len() >= FrameMeta::SIZE,
        "payload too short for FrameMeta"
    );

    let meta = FrameMeta::ref_from_bytes(&payload[..FrameMeta::SIZE])
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let width = meta.width;
    let height = meta.height;
    let stride = meta.stride;
    let num_regions = meta.num_regions as usize;

    let regions_end = FrameMeta::SIZE + WireRegion::SIZE * num_regions;
    anyhow::ensure!(
        payload.len() >= regions_end,
        "payload too short for region descriptors"
    );

    let mut regions = Vec::with_capacity(num_regions);
    let mut region_offsets = Vec::with_capacity(num_regions);
    let mut region_sizes = Vec::with_capacity(num_regions);
    let mut data_offset = 0usize;

    for i in 0..num_regions {
        let off = FrameMeta::SIZE + i * WireRegion::SIZE;
        let wr = WireRegion::ref_from_bytes(&payload[off..off + WireRegion::SIZE])
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        regions.push(DamageRect::new(wr.x, wr.y, wr.w, wr.h));
        region_offsets.push(data_offset);
        let cs = wr.compressed_size as usize;
        region_sizes.push(cs);
        data_offset += cs;
    }

    let compressed_data = payload[regions_end..].to_vec();
    anyhow::ensure!(
        compressed_data.len() >= data_offset,
        "payload too short for compressed data"
    );

    let compressed = CompressedFrame {
        data: compressed_data,
        region_offsets,
        region_sizes,
        stats: Default::default(),
    };

    Ok((width, height, stride, regions, compressed))
}

/// Create an `EventSender` callback that serializes input events and sends them via transport.
pub fn make_input_sender(sender: TransportSender) -> remoteway_input::capture_thread::EventSender {
    Box::new(move |event: &InputEvent| {
        let hdr = FrameHeader::new(
            0,
            MsgType::InputEvent,
            flags::LAST_CHUNK,
            size_of::<InputEvent>() as u32,
            0,
        );
        let mut wire = Vec::with_capacity(size_of::<FrameHeader>() + size_of::<InputEvent>());
        wire.extend_from_slice(hdr.as_bytes());
        wire.extend_from_slice(event.as_bytes());
        sender.send_input(wire)
    })
}

/// Main receive + decompress + display loop.
///
/// Runs as a tokio task. Receives compressed frames from transport, decompresses them,
/// optionally interpolates, and sends to the display thread.
pub async fn recv_decompress_loop(
    transport: &mut remoteway_transport::ssh_transport::SshTransport,
    mut display: DisplayThread,
    interpolation: Option<InterpolationManager>,
    shutdown: Arc<AtomicBool>,
) {
    let mut previous_frame: Vec<u8> = Vec::new();
    let mut has_previous = false;
    let mut interpolation = interpolation;
    let mut frame_count: u64 = 0;

    while !shutdown.load(Ordering::Acquire) {
        let msg = match transport.recv().await {
            Some(m) => m,
            None => {
                debug!("transport disconnected");
                shutdown.store(true, Ordering::Release);
                break;
            }
        };

        match msg.header.msg_type() {
            Ok(MsgType::AnchorFrame) | Ok(MsgType::FrameUpdate) => {
                let is_anchor = matches!(msg.header.msg_type(), Ok(MsgType::AnchorFrame));
                info!(
                    "received {}: {} bytes",
                    if is_anchor {
                        "AnchorFrame"
                    } else {
                        "FrameUpdate"
                    },
                    msg.payload.len(),
                );

                let parsed = match deserialize_frame_payload(&msg.payload) {
                    Ok(p) => p,
                    Err(e) => {
                        warn!("failed to parse frame payload: {e}");
                        continue;
                    }
                };

                let (width, height, stride, regions, compressed) = parsed;
                let frame_size = (stride * height) as usize;

                // Anchor frames reset the delta base to zeroes.
                if is_anchor || !has_previous {
                    previous_frame.resize(frame_size, 0);
                    if let Some(ref mut im) = interpolation {
                        im.reset_anchor();
                    }
                }

                let mut output = Vec::new();
                if let Err(e) = decompress_frame(
                    &compressed,
                    &previous_frame,
                    stride as usize,
                    &regions,
                    &mut output,
                ) {
                    warn!("decompress failed: {e}");
                    continue;
                }

                // Feed interpolation manager (if enabled).
                if let Some(ref mut im) = interpolation {
                    let gpu_frame = GpuFrame::from_data(
                        output.clone(),
                        width,
                        height,
                        stride,
                        msg.header.timestamp_ns,
                    );
                    im.push_frame(gpu_frame);
                }

                // Save as previous frame for next delta decode BEFORE moving into display.
                previous_frame.clear();
                previous_frame.extend_from_slice(&output);
                has_previous = true;

                // Convert DamageRect to display DamageRegion.
                let display_damage: Vec<remoteway_display::DamageRegion> = regions
                    .iter()
                    .map(|r| remoteway_display::DamageRegion {
                        x: r.x,
                        y: r.y,
                        width: r.width,
                        height: r.height,
                    })
                    .collect();

                // Move output into display frame (no extra clone).
                let display_frame = DisplayFrame {
                    surface_id: { msg.header.stream_id },
                    data: output,
                    damage: display_damage,
                    width,
                    height,
                    stride,
                    timestamp_ns: msg.header.timestamp_ns,
                };

                if !display.send_frame(display_frame) {
                    debug!("display queue full, frame dropped");
                }
                frame_count += 1;
                debug!(frame = frame_count, "frame displayed");
            }
            Ok(MsgType::Handshake) => {
                debug!("received server handshake (late)");
            }
            Ok(MsgType::CursorMove) => {
                debug!("cursor move received");
            }
            Ok(MsgType::Resize) => {
                if msg.payload.len() >= ResizePayload::SIZE
                    && let Ok(resize) =
                        ResizePayload::ref_from_bytes(&msg.payload[..ResizePayload::SIZE])
                {
                    let sid = resize.surface_id;
                    let w = resize.width;
                    let h = resize.height;
                    info!(
                        surface = sid,
                        width = w,
                        height = h,
                        "server resize notification"
                    );
                    has_previous = false;
                    if let Some(ref mut im) = interpolation {
                        im.clear();
                    }
                }
            }
            Ok(MsgType::Clipboard) => {
                debug!(len = msg.payload.len(), "clipboard data received");
            }
            Ok(other) => {
                debug!(?other, "ignoring message type");
            }
            Err(unknown) => {
                warn!(%unknown, "unknown message type");
            }
        }
    }

    display.stop();
}

#[cfg(test)]
mod tests {
    use remoteway_compress::delta::DamageRect;
    use remoteway_compress::pipeline::compress_frame;
    use remoteway_proto::frame::{FrameMeta, WireRegion};

    use super::*;

    fn make_frame(w: usize, h: usize) -> Vec<u8> {
        (0..w * h * 4).map(|i| (i * 7 + 3) as u8).collect()
    }

    /// Build wire payload the same way the server does.
    fn serialize_payload(
        width: u32,
        height: u32,
        stride: u32,
        regions: &[DamageRect],
        compressed: &CompressedFrame,
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

    #[test]
    fn deserialize_matches_server_serialize() {
        let w = 16u32;
        let h = 16u32;
        let stride = w * 4;
        let current = make_frame(w as usize, h as usize);
        let previous = vec![0u8; current.len()];
        let regions = vec![DamageRect::new(0, 0, w, h)];

        let compressed = compress_frame(&current, &previous, stride as usize, &regions);
        let payload = serialize_payload(w, h, stride, &regions, &compressed);

        let (dw, dh, ds, dregs, dc) = deserialize_frame_payload(&payload).unwrap();
        assert_eq!(dw, w);
        assert_eq!(dh, h);
        assert_eq!(ds, stride);
        assert_eq!(dregs.len(), 1);
        assert_eq!(dregs[0], regions[0]);
        assert_eq!(dc.region_sizes, compressed.region_sizes);
        assert_eq!(dc.data, compressed.data);
    }

    #[test]
    fn deserialize_empty_payload_errors() {
        assert!(deserialize_frame_payload(&[]).is_err());
    }

    #[test]
    fn deserialize_truncated_regions_errors() {
        let meta = FrameMeta::new(100, 100, 400, 5);
        let payload = meta.as_bytes().to_vec();
        assert!(deserialize_frame_payload(&payload).is_err());
    }

    #[test]
    fn full_compress_decompress_round_trip() {
        let w = 16u32;
        let h = 16u32;
        let stride = w * 4;
        let current = make_frame(w as usize, h as usize);
        let previous = vec![0u8; current.len()];
        let regions = vec![DamageRect::new(0, 0, w, h)];

        let compressed = compress_frame(&current, &previous, stride as usize, &regions);
        let payload = serialize_payload(w, h, stride, &regions, &compressed);

        let (_, _, ds, dregs, dc) = deserialize_frame_payload(&payload).unwrap();
        let mut output = Vec::new();
        decompress_frame(&dc, &previous, ds as usize, &dregs, &mut output).unwrap();
        assert_eq!(output, current);
    }

    #[test]
    fn deserialize_multi_region() {
        let w = 32u32;
        let h = 32u32;
        let stride = w * 4;
        let current = make_frame(w as usize, h as usize);
        let previous = vec![0u8; current.len()];
        let regions = vec![
            DamageRect::new(0, 0, 16, 16),
            DamageRect::new(16, 16, 16, 16),
        ];

        let compressed = compress_frame(&current, &previous, stride as usize, &regions);
        let payload = serialize_payload(w, h, stride, &regions, &compressed);

        let (dw, dh, ds, dregs, dc) = deserialize_frame_payload(&payload).unwrap();
        assert_eq!(dw, w);
        assert_eq!(dh, h);
        assert_eq!(ds, stride);
        assert_eq!(dregs.len(), 2);
        assert_eq!(dc.region_sizes.len(), 2);
    }

    #[test]
    fn deserialize_zero_regions() {
        let meta = FrameMeta::new(1920, 1080, 7680, 0);
        let payload = meta.as_bytes().to_vec();
        let (w, h, s, regions, cf) = deserialize_frame_payload(&payload).unwrap();
        assert_eq!(w, 1920);
        assert_eq!(h, 1080);
        assert_eq!(s, 7680);
        assert!(regions.is_empty());
        assert!(cf.data.is_empty());
    }

    #[test]
    fn deserialize_truncated_compressed_data_errors() {
        let meta = FrameMeta::new(100, 100, 400, 1);
        // WireRegion says 9999 bytes of compressed data, but payload has none.
        let wr = WireRegion::new(0, 0, 100, 100, 9999);
        let mut payload = Vec::new();
        payload.extend_from_slice(meta.as_bytes());
        payload.extend_from_slice(wr.as_bytes());
        // No compressed data appended.
        assert!(deserialize_frame_payload(&payload).is_err());
    }

    #[test]
    fn build_target_resolution_format() {
        let data = build_target_resolution(1920, 1080);
        assert_eq!(data.len(), 24); // FrameHeader(16) + TargetResolutionPayload(8)

        let hdr = FrameHeader::ref_from_bytes(&data[..16]).unwrap();
        assert_eq!(hdr.msg_type().unwrap(), MsgType::TargetResolution);
        assert_eq!({ hdr.flags }, flags::LAST_CHUNK);
        assert_eq!({ hdr.payload_len }, 8);

        use remoteway_proto::target_resolution::TargetResolutionPayload;
        let p = TargetResolutionPayload::ref_from_bytes(&data[16..24]).unwrap();
        assert_eq!({ p.width }, 1920);
        assert_eq!({ p.height }, 1080);
    }

    #[test]
    fn build_handshake_format() {
        let data = build_handshake();
        assert_eq!(data.len(), 24);

        let hdr = FrameHeader::ref_from_bytes(&data[..16]).unwrap();
        assert_eq!(hdr.msg_type().unwrap(), MsgType::Handshake);
        assert_eq!({ hdr.flags }, flags::LAST_CHUNK);
        assert_eq!({ hdr.payload_len }, 8);
    }

    #[test]
    fn input_event_wire_format() {
        let event = InputEvent::key(remoteway_proto::input::KeyEvent { key: 30, state: 1 });
        assert_eq!(event.as_bytes().len(), 16);

        // Verify that FrameHeader + InputEvent would produce correct wire size.
        let hdr = FrameHeader::new(0, MsgType::InputEvent, flags::LAST_CHUNK, 16, 0);
        let mut wire = Vec::new();
        wire.extend_from_slice(hdr.as_bytes());
        wire.extend_from_slice(event.as_bytes());
        assert_eq!(wire.len(), 32); // 16 header + 16 payload
    }

    #[test]
    fn deserialize_large_frame_4k() {
        let w = 3840u32;
        let h = 2160u32;
        let stride = w * 4;
        let current: Vec<u8> = (0..w as usize * h as usize * 4)
            .map(|i| (i * 3) as u8)
            .collect();
        let previous = vec![0u8; current.len()];
        let regions = vec![DamageRect::new(0, 0, w, h)];

        let compressed = compress_frame(&current, &previous, stride as usize, &regions);
        let payload = serialize_payload(w, h, stride, &regions, &compressed);

        let (dw, dh, ds, dregs, dc) = deserialize_frame_payload(&payload).unwrap();
        assert_eq!(dw, w);
        assert_eq!(dh, h);
        assert_eq!(ds, stride);
        assert_eq!(dregs.len(), 1);

        let mut output = Vec::new();
        decompress_frame(&dc, &previous, ds as usize, &dregs, &mut output).unwrap();
        assert_eq!(output, current);
    }

    #[test]
    fn deserialize_partial_meta_errors() {
        // Less than FrameMeta::SIZE bytes.
        let short = vec![0u8; FrameMeta::SIZE - 1];
        assert!(deserialize_frame_payload(&short).is_err());
    }

    #[test]
    fn build_target_resolution_zero_resets() {
        let data = build_target_resolution(0, 0);
        use remoteway_proto::target_resolution::TargetResolutionPayload;
        let p = TargetResolutionPayload::ref_from_bytes(&data[16..24]).unwrap();
        assert_eq!({ p.width }, 0);
        assert_eq!({ p.height }, 0);
    }

    #[test]
    fn sequential_delta_frames_round_trip() {
        let w = 8u32;
        let h = 8u32;
        let stride = w * 4;
        let regions = vec![DamageRect::new(0, 0, w, h)];

        let mut previous = vec![0u8; (stride * h) as usize];

        for seed in 0u8..5 {
            let current: Vec<u8> = (0..w as usize * h as usize * 4)
                .map(|i| ((i as u64 * (seed as u64 + 1) * 7 + 3) % 256) as u8)
                .collect();

            let compressed = compress_frame(&current, &previous, stride as usize, &regions);
            let payload = serialize_payload(w, h, stride, &regions, &compressed);
            let (_, _, ds, dregs, dc) = deserialize_frame_payload(&payload).unwrap();

            let mut output = Vec::new();
            decompress_frame(&dc, &previous, ds as usize, &dregs, &mut output).unwrap();
            assert_eq!(output, current, "mismatch at frame {seed}");

            previous = output;
        }
    }
}

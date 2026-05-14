use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anyhow::Result;
use remoteway_compress::delta::DamageRect;
use remoteway_compress::pipeline::{CompressedFrame, decompress_frame_into};
#[cfg(test)]
use remoteway_compress::pipeline::decompress_frame;
use remoteway_display::{DisplayFrame, DisplayThread};
use remoteway_interpolate::{GpuFrame, InterpolationManager};
use remoteway_proto::frame::{FrameMeta, WireRegion};
use remoteway_proto::header::{FrameHeader, MsgType, flags};
use remoteway_proto::input::InputEvent;
use remoteway_proto::resize::ResizePayload;
use remoteway_transport::ssh_transport::TransportSender;
use tracing::{debug, info, warn};
use zerocopy::{FromBytes, IntoBytes};

use crate::fps_overlay;

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
    upscale_factor: f64,
    debug: bool,
    compressor_kind: remoteway_compress::compressor::CompressorKind,
) {
    let mut previous_frame: Vec<u8> = Vec::new();
    let mut has_previous = false;
    let mut interpolation = interpolation;
    let mut frame_count: u64 = 0;
    // Reusable delta scratch for decompress — output buffer is freshly
    // allocated each frame because it is moved into DisplayFrame.
    let mut delta_scratch: Vec<u8> = Vec::new();
    // FPS counter — EMA over instantaneous inter-frame intervals.
    // Counts real frames only; interpolated frames inherit the same value.
    // First sample seeds the EMA so the very first overlay is meaningful.
    let mut last_frame_at: Option<Instant> = None;
    let mut fps_ema: f32 = 0.0;
    const FPS_ALPHA: f32 = 0.15;

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

                let mut output: Vec<u8> = Vec::new();
                if let Err(e) = decompress_frame_into(
                    &compressed,
                    &previous_frame,
                    stride as usize,
                    &regions,
                    &mut delta_scratch,
                    &mut output,
                    compressor_kind,
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

                // Try to generate an interpolated frame at the temporal midpoint.
                let interpolated = interpolation.as_mut().and_then(|im| {
                    im.interpolate(0.5).ok().flatten()
                });

                // Save as previous frame for next delta decode BEFORE moving into display.
                previous_frame.clear();
                previous_frame.extend_from_slice(&output);
                has_previous = true;

                // Spatial upscaling: GPU backend if available, else CPU bicubic.
                let do_upscale = (upscale_factor - 1.0).abs() > f64::EPSILON;
                let (display_w, display_h, display_stride, display_data) = if do_upscale {
                    let tw = (width as f64 * upscale_factor).round() as u32;
                    let th = (height as f64 * upscale_factor).round() as u32;

                    // Try GPU upscale via interpolation backend.
                    let gpu_upscaled = interpolation.as_ref().and_then(|im| {
                        let src = GpuFrame::from_data(
                            output.clone(),
                            width,
                            height,
                            stride,
                            msg.header.timestamp_ns,
                        );
                        {
                            let r = im.backend().upscale(&src, tw, th);
                            if let Err(ref e) = r { debug!("gpu upscale failed: {}", e); }
                            r.ok()
                        }
                    });

                    if let Some(gpu_frame) = gpu_upscaled {
                        debug!(
                            src = format!("{}x{}", width, height),
                            dst = format!("{}x{}", tw, th),
                            backend = "gpu",
                            "upscaling frame"
                        );
                        (
                            gpu_frame.width,
                            gpu_frame.height,
                            gpu_frame.stride,
                            gpu_frame.data,
                        )
                    } else {
                        debug!(
                            src = format!("{}x{}", width, height),
                            dst = format!("{}x{}", tw, th),
                            backend = "cpu",
                            "upscaling frame"
                        );
                        let mut upscaled = Vec::new();
                        let (uw, uh, us) =
                            upscale_bicubic(&output, width, height, stride, tw, th, &mut upscaled);
                        (uw, uh, us, upscaled)
                    }
                } else {
                    (width, height, stride, output)
                };

                // Convert DamageRect to display DamageRegion, scaling if needed.
                let scale_x = display_w as f64 / width as f64;
                let scale_y = display_h as f64 / height as f64;
                let mut display_damage: Vec<remoteway_display::DamageRegion> = regions
                    .iter()
                    .map(|r| remoteway_display::DamageRegion {
                        x: (r.x as f64 * scale_x).round() as u32,
                        y: (r.y as f64 * scale_y).round() as u32,
                        width: (r.width as f64 * scale_x).round() as u32,
                        height: (r.height as f64 * scale_y).round() as u32,
                    })
                    .collect();

                // Update FPS EMA from the inter-frame interval. Done before
                // the overlay is drawn so the displayed number reflects the
                // current frame's arrival, not the previous one.
                let now = Instant::now();
                if let Some(prev) = last_frame_at {
                    let dt = now.duration_since(prev).as_secs_f32();
                    if dt > 0.0 {
                        let instant_fps = 1.0 / dt;
                        fps_ema = if fps_ema == 0.0 {
                            instant_fps
                        } else {
                            fps_ema * (1.0 - FPS_ALPHA) + instant_fps * FPS_ALPHA
                        };
                    }
                }
                last_frame_at = Some(now);

                // Paint the FPS readout directly on the RGBA frame buffer
                // and add its rectangle to the damage list so the
                // compositor actually re-uploads that region.
                let mut display_data = display_data;
                if debug {
                    fps_overlay::draw_fps(
                        &mut display_data,
                        display_w,
                        display_h,
                        display_stride,
                        fps_ema,
                    );
                    let (ox, oy, ow, oh) = fps_overlay::overlay_rect(10, 10);
                    display_damage.push(remoteway_display::DamageRegion {
                        x: ox,
                        y: oy,
                        width: ow.min(display_w.saturating_sub(ox)),
                        height: oh.min(display_h.saturating_sub(oy)),
                    });
                }

                let display_frame = DisplayFrame {
                    surface_id: { msg.header.stream_id },
                    data: display_data,
                    damage: display_damage,
                    width: display_w,
                    height: display_h,
                    stride: display_stride,
                    timestamp_ns: msg.header.timestamp_ns,
                };

                if !display.send_frame(display_frame) {
                    debug!("display queue full, frame dropped");
                }
                frame_count += 1;
                debug!(frame = frame_count, "frame displayed");

                // If an interpolated frame was generated, upscale and display it.
                if let Some(interp_frame) = interpolated {
                    let (iw, ih, istride, idata) = if do_upscale {
                        let tw = (interp_frame.width as f64 * upscale_factor).round() as u32;
                        let th = (interp_frame.height as f64 * upscale_factor).round() as u32;
                        let gpu_up = interpolation.as_ref().and_then(|im| {
                            im.backend().upscale(&interp_frame, tw, th).ok()
                        });
                        if let Some(gpu) = gpu_up {
                            (gpu.width, gpu.height, gpu.stride, gpu.data)
                        } else {
                            let mut upscaled = Vec::new();
                            let (uw, uh, us) = upscale_bicubic(
                                &interp_frame.data,
                                interp_frame.width,
                                interp_frame.height,
                                interp_frame.stride,
                                tw, th,
                                &mut upscaled,
                            );
                            (uw, uh, us, upscaled)
                        }
                    } else {
                        (interp_frame.width, interp_frame.height, interp_frame.stride, interp_frame.data)
                    };

                    let full_damage = vec![remoteway_display::DamageRegion {
                        x: 0, y: 0,
                        width: iw,
                        height: ih,
                    }];

                    // Re-paint the FPS overlay so it doesn't flicker between
                    // the real frame and the synthesized one (which would
                    // otherwise carry whatever the GPU upscale produced under
                    // the badge area).
                    let mut idata = idata;
                    if debug {
                        fps_overlay::draw_fps(&mut idata, iw, ih, istride, fps_ema);
                    }

                    if !display.send_frame(DisplayFrame {
                        surface_id: msg.header.stream_id,
                        data: idata,
                        damage: full_damage,
                        width: iw,
                        height: ih,
                        stride: istride,
                        timestamp_ns: interp_frame.timestamp_ns,
                    }) {
                        debug!("interpolated frame dropped (display queue full)");
                    }
                }
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

// ---------------------------------------------------------------------------
// Client-side upscaling (Catmull-Rom bicubic interpolation)
// ---------------------------------------------------------------------------

/// Upscale a frame from `src_w×src_h` to `dst_w×dst_h` using Catmull-Rom bicubic filtering.
///
/// Uses a 4×4 kernel (16 samples per output pixel) for sharp, high-quality results.
/// Writes into `dst` (cleared and resized internally). Returns `(new_width, new_height, new_stride)`.
pub fn upscale_bicubic(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    src_stride: u32,
    dst_w: u32,
    dst_h: u32,
    dst: &mut Vec<u8>,
) -> (u32, u32, u32) {
    let dst_stride = dst_w * 4;
    let total = (dst_stride * dst_h) as usize;
    dst.clear();
    dst.resize(total, 0);

    /// Catmull-Rom cubic weight for sample offset `t` (0 = center-left, 1 = center-right).
    #[inline]
    fn cubic_weight(t: f64) -> [f64; 4] {
        let t2 = t * t;
        let t3 = t2 * t;
        [
            0.5 * (-t3 + 2.0 * t2 - t),        // sample at offset -1
            0.5 * (3.0 * t3 - 5.0 * t2 + 2.0), // sample at offset 0
            0.5 * (-3.0 * t3 + 4.0 * t2 + t),  // sample at offset +1
            0.5 * (t3 - t2),                   // sample at offset +2
        ]
    }

    for dy in 0..dst_h {
        let sy = (dy as f64 + 0.5) * src_h as f64 / dst_h as f64 - 0.5;
        let sy_floor = (sy.floor() as i32).max(0).min(src_h as i32 - 1);
        let fy = sy - sy.floor();

        // Clamp source row indices for bicubic kernel.
        let sy0 = (sy_floor - 1).max(0) as u32;
        let sy1 = sy_floor as u32;
        let sy2 = (sy_floor + 1).min(src_h as i32 - 1) as u32;
        let sy3 = (sy_floor + 2).min(src_h as i32 - 1) as u32;

        let wy = cubic_weight(fy);
        let dst_row = (dy * dst_stride) as usize;

        for dx in 0..dst_w {
            let sx = (dx as f64 + 0.5) * src_w as f64 / dst_w as f64 - 0.5;
            let sx_floor = (sx.floor() as i32).max(0).min(src_w as i32 - 1);
            let fx = sx - sx.floor();

            let sx0 = (sx_floor - 1).max(0) as u32;
            let sx1 = sx_floor as u32;
            let sx2 = (sx_floor + 1).min(src_w as i32 - 1) as u32;
            let sx3 = (sx_floor + 2).min(src_w as i32 - 1) as u32;

            let wx = cubic_weight(fx);
            let di = dst_row + (dx * 4) as usize;

            for c in 0..4 {
                // Convolve 4×4 kernel.
                let mut acc = 0.0f64;
                for (ky, &row_idx) in [sy0, sy1, sy2, sy3].iter().enumerate() {
                    let row = (row_idx * src_stride) as usize;
                    let sx = [sx0, sx1, sx2, sx3];
                    for (kx, &col_idx) in sx.iter().enumerate() {
                        let si = row + (col_idx * 4) as usize + c;
                        acc += f64::from(src[si]) * wx[kx] * wy[ky];
                    }
                }
                dst[di + c] = acc.round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    (dst_w, dst_h, dst_stride)
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

    // ── Upscaling tests ──────────────────────────────────────────────────

    #[test]
    fn upscale_same_size_is_copy() {
        let src: Vec<u8> = (0..16).map(|i| i as u8).collect();
        let mut dst = Vec::new();
        let (w, h, s) = upscale_bicubic(&src, 2, 2, 8, 2, 2, &mut dst);
        assert_eq!((w, h, s), (2, 2, 8));
        assert_eq!(dst, src);
    }

    #[test]
    fn upscale_2x_bicubic() {
        let src = vec![255u8; 2 * 2 * 4];
        let mut dst = Vec::new();
        let (w, h, s) = upscale_bicubic(&src, 2, 2, 8, 4, 4, &mut dst);
        assert_eq!((w, h, s), (4, 4, 16));
        assert_eq!(dst.len(), 4 * 4 * 4);
        // Bicubic may over/undershoot at edges — allow small deviation.
        for &px in &dst {
            assert!(px >= 240, "px={px}");
        }
    }

    #[test]
    fn upscale_preserves_distinct_pixels() {
        let mut src = Vec::new();
        src.extend_from_slice(&[255u8, 0, 0, 255]);
        src.extend_from_slice(&[0, 0, 255, 255]);
        let mut dst = Vec::new();
        upscale_bicubic(&src, 2, 1, 8, 4, 1, &mut dst);
        // Leftmost should stay red-ish, rightmost blue-ish
        assert!(dst[0] > 200);
        assert!(dst[2] < 50);
        assert!(dst[12] < 50);
        assert!(dst[14] > 200);
    }

    #[test]
    fn upscale_min_dimensions() {
        let src = vec![128u8; 4];
        let mut dst = Vec::new();
        let (w, h, _) = upscale_bicubic(&src, 1, 1, 4, 8, 8, &mut dst);
        assert_eq!((w, h), (8, 8));
        assert_eq!(dst.len(), 8 * 8 * 4);
    }
}

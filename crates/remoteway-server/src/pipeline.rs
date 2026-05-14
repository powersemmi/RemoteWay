use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;

use anyhow::{Context, Result};
use remoteway_capture::backend::CaptureBackend;
use remoteway_capture::ext_capture::{CaptureSource, ExtImageCaptureBackend};
use remoteway_capture::screencopy::WlrScreencopyBackend;
use remoteway_capture::thread::CaptureThread;
use remoteway_compress::delta::DamageRect;
use remoteway_compress::pipeline::{CompressedFrame, compress_frame_into};
use remoteway_core::thread_config::ThreadConfig;
use remoteway_input::inject_thread::InputInjectThread;
use remoteway_proto::frame::{FrameMeta, WireRegion};
use remoteway_proto::header::{FrameHeader, MsgType, flags};
use remoteway_proto::input::InputEvent;
use remoteway_transport::chunk_sender;
use remoteway_transport::ssh_transport::TransportSender;
use tracing::{debug, info, warn};
use zerocopy::{FromBytes, IntoBytes};

use crate::cli::{CaptureBackendArg, CompressArg, DownscaleFilterArg};

/// Maximum chunk size for frame data on the wire (64 KiB).
const CHUNK_SIZE: usize = 64 * 1024;

/// Send an anchor frame every N frames to prevent error accumulation
/// and enable resync after packet loss.
const ANCHOR_INTERVAL: u64 = 300;

/// Cut a full `width × height` rect into roughly equal horizontal stripes so
/// the compress pipeline can fan out across CPU cores.
///
/// Returns `n` non-overlapping `DamageRect`s that together cover the whole
/// frame. The last stripe absorbs the remainder if `height` is not divisible
/// by `n`. `n` is clamped to 1 when `height < n` (one-pixel-tall stripes
/// would just hurt cache locality without parallelism gain).
fn stripe_full_frame(width: u32, height: u32, n: usize) -> Vec<DamageRect> {
    let n = (n as u32).min(height).max(1);
    if n == 1 {
        return vec![DamageRect::new(0, 0, width, height)];
    }
    let base = height / n;
    let rem = height % n;
    let mut stripes = Vec::with_capacity(n as usize);
    let mut y = 0u32;
    for i in 0..n {
        let h = base + if i < rem { 1 } else { 0 };
        stripes.push(DamageRect::new(0, y, width, h));
        y += h;
    }
    stripes
}

/// Nearest-neighbor downscale of an RGBA frame.
///
/// `src` is the source pixel data, `src_w × src_h` at `src_stride` bytes per row.
/// Writes into `dst` (cleared and resized internally). Returns the destination stride.
///
/// Picks a single source pixel per destination pixel — fast but causes severe
/// aliasing on non-axis-aligned content and destroys the GPU's anti-aliasing.
/// Prefer `downscale_box` unless CPU budget is the binding constraint.
fn downscale_nearest(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    src_stride: u32,
    dst_w: u32,
    dst_h: u32,
    dst: &mut Vec<u8>,
) -> u32 {
    let dst_stride = dst_w * 4;
    let total = (dst_stride * dst_h) as usize;
    dst.clear();
    dst.resize(total, 0);

    for dy in 0..dst_h {
        let sy = ((dy as u64 * src_h as u64) / dst_h as u64) as u32;
        let src_row = (sy * src_stride) as usize;
        let dst_row = (dy * dst_stride) as usize;
        for dx in 0..dst_w {
            let sx = ((dx as u64 * src_w as u64) / dst_w as u64) as u32;
            let si = src_row + (sx * 4) as usize;
            let di = dst_row + (dx * 4) as usize;
            dst[di..di + 4].copy_from_slice(&src[si..si + 4]);
        }
    }

    dst_stride
}

/// Area-weighted (box-filter) downscale of an RGBA frame.
///
/// Each destination pixel is the unweighted average of every source pixel in
/// the half-open rectangle `[sx0, sx1) × [sy0, sy1)` that the destination
/// pixel covers, with the bounds derived from the integer ratio
/// `dst_w / src_w × dst_h / src_h`. This preserves the energy/anti-aliasing
/// already present in the source — critical for keeping text strokes and
/// thin UI lines intact through the wire so the client-side FSR upscaler
/// has something coherent to reconstruct from.
///
/// For 0.5× scale every destination pixel averages a clean 2×2 block; for
/// non-integer ratios some destination pixels cover 2×2 and others 3×2 / 3×3,
/// which is correct (it varies by sub-pixel alignment).
///
/// Parallelised across destination rows via rayon — at 2560×1440 → 1280×720
/// the serial loop is ~8–12 ms on a single core (CPU-bound on the inner
/// 2×2 average), enough to cap the pipeline at ~80 fps even before the
/// rest of the work; row-parallel scaling brings it under 1 ms on any
/// modern multi-core box.
fn downscale_box(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    src_stride: u32,
    dst_w: u32,
    dst_h: u32,
    dst: &mut Vec<u8>,
) -> u32 {
    use rayon::prelude::*;

    let dst_stride = dst_w * 4;
    let total = (dst_stride * dst_h) as usize;
    dst.clear();
    dst.resize(total, 0);

    if dst_w == 0 || dst_h == 0 {
        return dst_stride;
    }

    // Upscale (or 1:1) is not the responsibility of this function — caller
    // should skip downscaling entirely. Defensive guard: degrade to nearest
    // copy of (0,0) for safety. Won't be hit in practice.
    if dst_w >= src_w && dst_h >= src_h {
        return dst_stride;
    }

    // Split `dst` into one mutable slice per destination row so rayon can
    // dispatch each row to a different worker without aliasing. `src` is
    // shared by all workers as a read-only borrow.
    dst.par_chunks_mut(dst_stride as usize)
        .enumerate()
        .for_each(|(dy_us, row_buf)| {
            let dy = dy_us as u32;
            let sy0 = (u64::from(dy) * u64::from(src_h) / u64::from(dst_h)) as u32;
            let sy1_raw = (u64::from(dy + 1) * u64::from(src_h) / u64::from(dst_h)) as u32;
            let sy1 = sy1_raw.max(sy0 + 1).min(src_h);

            for dx in 0..dst_w {
                let sx0 = (u64::from(dx) * u64::from(src_w) / u64::from(dst_w)) as u32;
                let sx1_raw = (u64::from(dx + 1) * u64::from(src_w) / u64::from(dst_w)) as u32;
                let sx1 = sx1_raw.max(sx0 + 1).min(src_w);

                let mut acc_r: u32 = 0;
                let mut acc_g: u32 = 0;
                let mut acc_b: u32 = 0;
                let mut acc_a: u32 = 0;
                let mut count: u32 = 0;

                for sy in sy0..sy1 {
                    let row_off = (sy * src_stride) as usize;
                    for sx in sx0..sx1 {
                        let off = row_off + (sx * 4) as usize;
                        acc_r += u32::from(src[off]);
                        acc_g += u32::from(src[off + 1]);
                        acc_b += u32::from(src[off + 2]);
                        acc_a += u32::from(src[off + 3]);
                        count += 1;
                    }
                }

                // count >= 1 because we clamped sx1 >= sx0+1 and sy1 >= sy0+1.
                let di = (dx * 4) as usize;
                row_buf[di]     = (acc_r / count) as u8;
                row_buf[di + 1] = (acc_g / count) as u8;
                row_buf[di + 2] = (acc_b / count) as u8;
                row_buf[di + 3] = (acc_a / count) as u8;
            }
        });

    dst_stride
}

/// Detect and create the capture backend based on CLI args.
pub fn create_capture_backend(
    capture_arg: &CaptureBackendArg,
    output: Option<&str>,
    app_id: Option<&str>,
) -> Result<Box<dyn CaptureBackend>> {
    // Per-window capture path: ext-image-capture supports toplevel capture natively.
    if let Some(app_id) = app_id {
        return match capture_arg {
            CaptureBackendArg::WlrScreencopy => {
                anyhow::bail!(
                    "--app-id requires ext-image-capture or portal backend; \
                     wlr-screencopy does not support per-window capture"
                );
            }
            #[cfg(feature = "portal")]
            CaptureBackendArg::Portal => {
                return remoteway_capture::portal::PortalBackend::new()
                    .map(|b| Box::new(b) as Box<dyn CaptureBackend>)
                    .context("failed to create portal capture backend (window)");
            }
            CaptureBackendArg::Auto | CaptureBackendArg::ExtImageCapture => {
                remoteway_capture::detect::detect_toplevel_backend(app_id)
                    .context(format!("failed to capture toplevel '{app_id}'"))
            }
        };
    }

    // Output capture path.
    match capture_arg {
        CaptureBackendArg::Auto => remoteway_capture::detect::detect_backend(output)
            .context("failed to detect capture backend"),
        CaptureBackendArg::WlrScreencopy => WlrScreencopyBackend::new(output)
            .map(|b| Box::new(b) as Box<dyn CaptureBackend>)
            .context("failed to create wlr-screencopy backend"),
        CaptureBackendArg::ExtImageCapture => {
            let source = match output {
                Some(name) => CaptureSource::Output(Some(name.to_string())),
                None => CaptureSource::Output(None),
            };
            ExtImageCaptureBackend::new(source)
                .map(|b| Box::new(b) as Box<dyn CaptureBackend>)
                .context("failed to create ext-image-capture backend")
        }
        #[cfg(feature = "portal")]
        CaptureBackendArg::Portal => remoteway_capture::portal::PortalBackend::new()
            .map(|b| Box::new(b) as Box<dyn CaptureBackend>)
            .context("failed to create portal capture backend"),
    }
}

/// Detect capture backend for a just-spawned child process (diff-based).
///
/// Finds the first new toplevel window not present in `known_identifiers`.
pub fn create_capture_backend_for_child(
    capture_arg: &CaptureBackendArg,
    known_identifiers: &[String],
) -> Result<Box<dyn CaptureBackend>> {
    match capture_arg {
        CaptureBackendArg::WlrScreencopy => {
            anyhow::bail!(
                "per-window capture requires ext-image-capture or portal backend; \
                                 wlr-screencopy does not support it"
            );
        }
        #[cfg(feature = "portal")]
        CaptureBackendArg::Portal => remoteway_capture::portal::PortalBackend::new()
            .map(|b| Box::new(b) as Box<dyn CaptureBackend>)
            .context("failed to create portal capture backend (window)"),
        CaptureBackendArg::Auto | CaptureBackendArg::ExtImageCapture => {
            remoteway_capture::detect::detect_new_toplevel_backend(known_identifiers)
                .context("failed to detect child window")
        }
    }
}

/// Build the handshake payload describing server capabilities.
pub fn build_handshake(capture_arg: &CaptureBackendArg, compress_arg: &CompressArg) -> Vec<u8> {
    use remoteway_proto::handshake::{HandshakePayload, capture_flags, compress_flags};

    let cap = match capture_arg {
        CaptureBackendArg::Auto => {
            capture_flags::WLR_SCREENCOPY | capture_flags::EXT_IMAGE_CAPTURE | capture_flags::PORTAL
        }
        CaptureBackendArg::WlrScreencopy => capture_flags::WLR_SCREENCOPY,
        CaptureBackendArg::ExtImageCapture => capture_flags::EXT_IMAGE_CAPTURE,
        #[cfg(feature = "portal")]
        CaptureBackendArg::Portal => capture_flags::PORTAL,
    };

    let comp = match compress_arg {
        CompressArg::None => compress_flags::NONE,
        CompressArg::Lz4 => compress_flags::LZ4,
        CompressArg::Zstd => compress_flags::LZ4 | compress_flags::ZSTD,
    };

    let hs = HandshakePayload::new(cap, comp);
    let hdr = FrameHeader::new(0, MsgType::Handshake, flags::LAST_CHUNK, 8, 0);

    let mut wire = Vec::with_capacity(24);
    wire.extend_from_slice(hdr.as_bytes());
    wire.extend_from_slice(hs.as_bytes());
    wire
}

/// Serialize a compressed frame into wire payload bytes using zerocopy.
pub fn serialize_frame_payload(
    width: u32,
    height: u32,
    stride: u32,
    regions: &[DamageRect],
    compressed: &CompressedFrame,
) -> Vec<u8> {
    let meta = FrameMeta::new(width, height, stride, regions.len() as u32);
    let total = FrameMeta::SIZE + WireRegion::SIZE * regions.len() + compressed.data.len();
    let mut payload = Vec::with_capacity(total);

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

/// Compress+send pipeline thread: reads frames from capture, compresses, sends via transport.
///
/// Runs on a dedicated OS thread (Core 2, default scheduler).
pub fn compress_send_loop(
    mut capture: CaptureThread,
    sender: TransportSender,
    compress_arg: &CompressArg,
    shutdown: &AtomicBool,
    scale: f64,
    downscale_filter: DownscaleFilterArg,
) {
    let compressor_kind = compress_arg.to_kind();
    let mut previous_frame: Vec<u8> = Vec::new();
    let mut is_first = true;
    let mut frame_count: u64 = 0;
    let mut need_full_damage = false;
    let mut scaled_buf: Vec<u8> = Vec::new();
    let do_downscale = (scale - 1.0).abs() > f64::EPSILON;
    // Reusable scratch buffers for the compress pipeline.
    let mut delta_scratch: Vec<u8> = Vec::new();
    let mut compressed = CompressedFrame::default();

    while !shutdown.load(Ordering::Acquire) {
        // Drain the capture ring to the LATEST frame. Intermediate frames
        // are stale — processing them would only add pipeline latency.
        let mut frame = match capture.try_recv() {
            Some(f) => {
                f
            }
            None => {
                if capture.is_finished() {
                    info!("capture thread finished, signalling shutdown");
                    shutdown.store(true, Ordering::Release);
                    break;
                }
                // No frame ready: sleep briefly instead of spin-waiting.
                // 2 ms keeps latency low while yielding the CPU.
                std::thread::sleep(std::time::Duration::from_millis(2));
                continue;
            }
        };
        let mut skipped = 0u32;
        while let Some(newer) = capture.try_recv() {
            frame = newer;
            skipped += 1;
        }
        if skipped > 0 {
            // We skipped intermediate captures — their damage regions are
            // lost, so we must use full-frame damage for correctness.
            need_full_damage = true;
            debug!(skipped, "drained capture ring to latest frame");
        }

        // Check if the capture thread dropped frames (ring overflow).
        // If so, the compositor's damage is relative to a frame we never
        // processed, so we must use full-frame damage.
        if capture.take_dropped_flag() {
            need_full_damage = true;
        }

        // Apply server-side downscaling if scale < 1.0.
        let (frame_data, frame_w, frame_h, frame_stride): (&[u8], u32, u32, u32) = if do_downscale {
            let tw = (frame.width as f64 * scale).round() as u32;
            let th = (frame.height as f64 * scale).round() as u32;
            let t0 = std::time::Instant::now();
            let _ = match downscale_filter {
                DownscaleFilterArg::Box => downscale_box(
                    &frame.data,
                    frame.width,
                    frame.height,
                    frame.stride,
                    tw,
                    th,
                    &mut scaled_buf,
                ),
                DownscaleFilterArg::Nearest => downscale_nearest(
                    &frame.data,
                    frame.width,
                    frame.height,
                    frame.stride,
                    tw,
                    th,
                    &mut scaled_buf,
                ),
            };
            debug!(
                src = format!("{}x{}", frame.width, frame.height),
                dst = format!("{tw}x{th}"),
                filter = ?downscale_filter,
                ms = t0.elapsed().as_secs_f32() * 1000.0,
                "downscale done"
            );
            (&scaled_buf, tw, th, tw * 4)
        } else {
            (&frame.data, frame.width, frame.height, frame.stride)
        };

        // Periodic anchor frames for resync / error accumulation prevention.
        let force_anchor =
            is_first || (frame_count > 0 && frame_count.is_multiple_of(ANCHOR_INTERVAL));

        let use_full_damage = force_anchor || frame.damage.is_empty() || need_full_damage;

        let regions: Vec<DamageRect> = if use_full_damage || do_downscale {
            // Full-frame path (anchor frame, drained captures, downscale,
            // or no compositor damage info). A single huge region would
            // pin LZ4/zstd to one core. Split horizontally into one stripe
            // per rayon worker so the per-region compress loop downstream
            // actually fans out across CPUs. The number of CPUs from rayon's
            // pool is the right scale — same pool also runs box-downscale.
            stripe_full_frame(frame_w, frame_h, rayon::current_num_threads().max(1))
        } else {
            frame
                .damage
                .iter()
                .map(|d| DamageRect::new(d.x, d.y, d.width, d.height))
                .collect()
        };

        let msg_type = if force_anchor {
            MsgType::AnchorFrame
        } else {
            MsgType::FrameUpdate
        };

        if force_anchor {
            previous_frame.resize(frame_data.len(), 0);
        }

        compress_frame_into(
            frame_data,
            &previous_frame,
            frame_stride as usize,
            &regions,
            &mut delta_scratch,
            &mut compressed,
            compressor_kind,
        );

        let payload =
            serialize_frame_payload(frame_w, frame_h, frame_stride, &regions, &compressed);

        // Frame on wire with chunking.
        let mut wire = Vec::new();
        chunk_sender::split_into_chunks(
            1, // stream_id = 1 for primary surface
            msg_type,
            flags::COMPRESSED,
            &payload,
            frame.timestamp_ns,
            CHUNK_SIZE,
            &mut wire,
        );

        let sent = if force_anchor {
            let r = sender.send_anchor(wire);
            info!(
                "anchor frame sent: {}x{} {}b",
                frame_w,
                frame_h,
                payload.len()
            );
            r
        } else {
            let r = sender.try_send_frame(wire);
            if r {
                debug!("frame sent: {}x{} {}b", frame_w, frame_h, payload.len());
            }
            r
        };

        if sent {
            // Only advance the delta base when the frame actually reaches the
            // client. Otherwise the server's previous_frame diverges from the
            // client's, causing cumulative artifacts.
            if do_downscale {
                // scaled_buf is reused next iteration — copy is required.
                previous_frame.clear();
                previous_frame.extend_from_slice(&scaled_buf);
            } else {
                // Zero-copy: move the capture buffer into previous_frame.
                // The old previous_frame's allocation goes back into `frame.data`,
                // so the capture thread keeps a sized buffer too.
                std::mem::swap(&mut previous_frame, &mut frame.data);
            }
            need_full_damage = false;
        } else {
            warn!(frame = frame_count, "frame dropped (backpressure)");
            // The client didn't receive this frame, so the next frame's damage
            // regions (relative to the capture that just happened) won't cover
            // all differences relative to the client's actual state.
            need_full_damage = true;
        }

        is_first = false;
        frame_count += 1;

        debug!(
            frame = frame_count,
            ratio = compressed.stats.compressed_bytes as f64
                / compressed.stats.original_bytes.max(1) as f64,
            "frame compressed"
        );
    }

    capture.stop();
}

/// Receive loop: reads incoming messages from transport, dispatches input events.
///
/// Runs as a tokio task on the control-plane runtime.
pub async fn recv_dispatch_loop(
    transport: &mut remoteway_transport::ssh_transport::SshTransport,
    mut input_inject: InputInjectThread,
    shutdown: Arc<AtomicBool>,
) {
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
            Ok(MsgType::InputEvent) => {
                if msg.payload.len() >= size_of::<InputEvent>()
                    && let Ok(event) =
                        InputEvent::ref_from_bytes(&msg.payload[..size_of::<InputEvent>()])
                    && !input_inject.send(*event)
                {
                    warn!("input inject queue full, event dropped");
                }
            }
            Ok(MsgType::Handshake) => {
                debug!("received client handshake");
            }
            Ok(other) => {
                debug!(?other, "ignoring unexpected message type");
            }
            Err(unknown) => {
                warn!(%unknown, "unknown message type");
            }
        }
    }

    input_inject.stop();
}

/// Spawn the compress+send thread, returning its join handle.
pub fn spawn_compress_thread(
    capture: CaptureThread,
    sender: TransportSender,
    compress_arg: CompressArg,
    shutdown: Arc<AtomicBool>,
    scale: f64,
    downscale_filter: DownscaleFilterArg,
) -> Result<JoinHandle<()>> {
    let config = ThreadConfig::new(2, 0, "compress-send");
    config
        .spawn(move || {
            compress_send_loop(
                capture,
                sender,
                &compress_arg,
                &shutdown,
                scale,
                downscale_filter,
            );
        })
        .context("failed to spawn compress-send thread")
}

#[cfg(test)]
mod tests {
    use super::*;
    use remoteway_compress::delta::DamageRect;
    use remoteway_compress::pipeline::{CompressedFrame, compress_frame};

    #[test]
    fn serialize_round_trip_single_region() {
        let w = 16u32;
        let h = 16u32;
        let stride = w * 4;
        let current: Vec<u8> = (0..w as usize * h as usize * 4)
            .map(|i| (i * 7 + 3) as u8)
            .collect();
        let previous = vec![0u8; current.len()];
        let regions = vec![DamageRect::new(0, 0, w, h)];

        let compressed = compress_frame(&current, &previous, stride as usize, &regions);
        let payload = serialize_frame_payload(w, h, stride, &regions, &compressed);

        // Deserialize via zerocopy.
        let meta = FrameMeta::ref_from_bytes(&payload[..FrameMeta::SIZE]).unwrap();
        assert_eq!({ meta.width }, w);
        assert_eq!({ meta.height }, h);
        assert_eq!({ meta.stride }, stride);
        assert_eq!({ meta.num_regions }, 1);

        let wr_off = FrameMeta::SIZE;
        let wr = WireRegion::ref_from_bytes(&payload[wr_off..wr_off + WireRegion::SIZE]).unwrap();
        assert_eq!({ wr.x }, 0);
        assert_eq!({ wr.w }, w);
        assert_eq!({ wr.compressed_size }, compressed.region_sizes[0] as u32);

        let data_off = wr_off + WireRegion::SIZE;
        assert_eq!(&payload[data_off..], &compressed.data[..]);
    }

    #[test]
    fn serialize_round_trip_multi_region() {
        let w = 32u32;
        let h = 32u32;
        let stride = w * 4;
        let current: Vec<u8> = (0..w as usize * h as usize * 4)
            .map(|i| (i * 11) as u8)
            .collect();
        let previous = vec![0u8; current.len()];
        let regions = vec![
            DamageRect::new(0, 0, 16, 16),
            DamageRect::new(16, 16, 16, 16),
        ];

        let compressed = compress_frame(&current, &previous, stride as usize, &regions);
        let payload = serialize_frame_payload(w, h, stride, &regions, &compressed);

        let meta = FrameMeta::ref_from_bytes(&payload[..FrameMeta::SIZE]).unwrap();
        assert_eq!({ meta.num_regions }, 2);

        let mut off = FrameMeta::SIZE;
        for (i, rect) in regions.iter().enumerate() {
            let wr = WireRegion::ref_from_bytes(&payload[off..off + WireRegion::SIZE]).unwrap();
            assert_eq!({ wr.x }, rect.x);
            assert_eq!({ wr.y }, rect.y);
            assert_eq!({ wr.w }, rect.width);
            assert_eq!({ wr.h }, rect.height);
            assert_eq!({ wr.compressed_size }, compressed.region_sizes[i] as u32);
            off += WireRegion::SIZE;
        }

        assert_eq!(&payload[off..], &compressed.data[..]);
    }

    #[test]
    fn serialize_empty_regions() {
        let compressed = CompressedFrame {
            data: vec![],
            region_offsets: vec![],
            region_sizes: vec![],
            stats: Default::default(),
        };
        let payload = serialize_frame_payload(1920, 1080, 7680, &[], &compressed);
        assert_eq!(payload.len(), FrameMeta::SIZE);
        let meta = FrameMeta::ref_from_bytes(&payload[..FrameMeta::SIZE]).unwrap();
        assert_eq!({ meta.num_regions }, 0);
        assert_eq!({ meta.width }, 1920);
    }

    #[test]
    fn build_handshake_auto_lz4() {
        let data = build_handshake(&CaptureBackendArg::Auto, &CompressArg::Lz4);
        assert_eq!(data.len(), 24); // FrameHeader(16) + HandshakePayload(8)

        let hdr = FrameHeader::ref_from_bytes(&data[..16]).unwrap();
        assert_eq!(hdr.msg_type().unwrap(), MsgType::Handshake);
        assert_eq!({ hdr.flags }, flags::LAST_CHUNK);
        assert_eq!({ hdr.payload_len }, 8);
    }

    #[test]
    fn build_handshake_zstd() {
        let data = build_handshake(&CaptureBackendArg::WlrScreencopy, &CompressArg::Zstd);
        assert_eq!(data.len(), 24);

        use remoteway_proto::handshake::{HandshakePayload, capture_flags, compress_flags};
        let hs = HandshakePayload::ref_from_bytes(&data[16..24]).unwrap();
        assert_eq!({ hs.capture_flags }, capture_flags::WLR_SCREENCOPY);
        assert_ne!({ hs.compress_flags } & compress_flags::ZSTD, 0);
    }

    #[test]
    fn build_handshake_ext_image_capture() {
        let data = build_handshake(&CaptureBackendArg::ExtImageCapture, &CompressArg::Lz4);
        use remoteway_proto::handshake::{HandshakePayload, capture_flags};
        let hs = HandshakePayload::ref_from_bytes(&data[16..24]).unwrap();
        assert_eq!({ hs.capture_flags }, capture_flags::EXT_IMAGE_CAPTURE);
    }

    #[test]
    fn downscale_box_halves_resolution_averages_2x2() {
        // 4x4 source: each row/col has a distinct value so the 2x2 average
        // for each destination pixel is a known integer.
        //
        //  row0: 10 30 50 70
        //  row1: 20 40 60 80
        //  row2: 90 110 130 150
        //  row3: 100 120 140 160
        //
        // Top-left dst = avg(10,30,20,40) = 25. Top-right dst = avg(50,70,60,80) = 65.
        // Bottom-left = avg(90,110,100,120) = 105. Bottom-right = avg(130,150,140,160) = 145.
        let pattern: [u8; 16] = [
            10, 30, 50, 70,
            20, 40, 60, 80,
            90, 110, 130, 150,
            100, 120, 140, 160,
        ];
        let (sw, sh) = (4u32, 4u32);
        let stride = sw * 4;
        let mut src = vec![0u8; (stride * sh) as usize];
        for y in 0..sh as usize {
            for x in 0..sw as usize {
                let off = y * stride as usize + x * 4;
                let v = pattern[y * 4 + x];
                src[off] = v;
                src[off + 1] = v;
                src[off + 2] = v;
                src[off + 3] = 255;
            }
        }

        let mut dst = Vec::new();
        let dst_stride = downscale_box(&src, sw, sh, stride, 2, 2, &mut dst);

        assert_eq!(dst_stride, 8);
        assert_eq!(dst.len(), 16);
        // R channel of each dst pixel == averaged greyscale value.
        assert_eq!(dst[0], 25);  // top-left
        assert_eq!(dst[4], 65);  // top-right
        assert_eq!(dst[8], 105); // bottom-left
        assert_eq!(dst[12], 145); // bottom-right
        // Alpha preserved at 255.
        assert_eq!(dst[3], 255);
        assert_eq!(dst[15], 255);
    }

    #[test]
    fn downscale_box_preserves_solid_color() {
        // A uniform color must come out identical — averaging Ns of the same
        // value is that value. This guards the integer-divide rounding path.
        let (sw, sh) = (8u32, 6u32);
        let stride = sw * 4;
        let src = vec![137u8; (stride * sh) as usize];
        let mut dst = Vec::new();
        downscale_box(&src, sw, sh, stride, 4, 3, &mut dst);
        assert!(dst.iter().all(|&b| b == 137));
    }

    #[test]
    fn downscale_nearest_halves_resolution() {
        let (sw, sh) = (4u32, 4u32);
        let stride = sw * 4;
        // Fill a 4x4 RGBA image with known pixel values.
        let mut src = vec![0u8; (stride * sh) as usize];
        for y in 0..sh {
            for x in 0..sw {
                let off = (y * stride + x * 4) as usize;
                src[off] = x as u8;
                src[off + 1] = y as u8;
                src[off + 2] = 0;
                src[off + 3] = 255;
            }
        }

        let (dw, dh) = (2u32, 2u32);
        let mut dst = Vec::new();
        let dst_stride = downscale_nearest(&src, sw, sh, stride, dw, dh, &mut dst);

        assert_eq!(dst_stride, dw * 4);
        assert_eq!(dst.len(), (dw * dh * 4) as usize);

        // Top-left pixel should be (0,0,...).
        assert_eq!(dst[0], 0);
        assert_eq!(dst[1], 0);
    }

    #[test]
    fn downscale_nearest_same_size_is_copy() {
        let (w, h) = (2u32, 2u32);
        let stride = w * 4;
        let src: Vec<u8> = (0..(w * h * 4) as u8).collect();

        let mut dst = Vec::new();
        let dst_stride = downscale_nearest(&src, w, h, stride, w, h, &mut dst);

        assert_eq!(dst_stride, stride);
        assert_eq!(dst, src);
    }

    #[test]
    fn downscale_nearest_upscale_not_applied() {
        // When target > source, compress_send_loop skips downscale,
        // but the function itself should still work (nearest neighbor stretch).
        let (sw, sh) = (2u32, 2u32);
        let stride = sw * 4;
        let src = vec![128u8; (stride * sh) as usize];

        let (dw, dh) = (4u32, 4u32);
        let mut dst = Vec::new();
        let dst_stride = downscale_nearest(&src, sw, sh, stride, dw, dh, &mut dst);

        assert_eq!(dst_stride, dw * 4);
        assert_eq!(dst.len(), (dw * dh * 4) as usize);
        // All pixels should be 128 since source is uniform.
        assert!(dst.iter().all(|&b| b == 128));
    }

    #[test]
    fn serialize_large_frame_4k() {
        let w = 3840u32;
        let h = 2160u32;
        let stride = w * 4;
        let current: Vec<u8> = (0..w as usize * h as usize * 4)
            .map(|i| (i * 3) as u8)
            .collect();
        let previous = vec![0u8; current.len()];
        let regions = vec![DamageRect::new(0, 0, w, h)];

        let compressed = compress_frame(&current, &previous, stride as usize, &regions);
        let payload = serialize_frame_payload(w, h, stride, &regions, &compressed);

        assert!(payload.len() > FrameMeta::SIZE + WireRegion::SIZE);
        let meta = FrameMeta::ref_from_bytes(&payload[..FrameMeta::SIZE]).unwrap();
        assert_eq!({ meta.width }, 3840);
        assert_eq!({ meta.height }, 2160);
    }
}

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::JoinHandle;

use anyhow::{Context, Result};
use remoteway_capture::backend::CaptureBackend;
use remoteway_capture::ext_capture::{CaptureSource, ExtImageCaptureBackend};
use remoteway_capture::screencopy::WlrScreencopyBackend;
use remoteway_capture::thread::CaptureThread;
use remoteway_compress::delta::DamageRect;
use remoteway_compress::pipeline::{CompressedFrame, compress_frame};
use remoteway_core::thread_config::ThreadConfig;
use remoteway_input::inject_thread::InputInjectThread;
use remoteway_proto::frame::{FrameMeta, WireRegion};
use remoteway_proto::header::{FrameHeader, MsgType, flags};
use remoteway_proto::input::InputEvent;
use remoteway_transport::chunk_sender;
use remoteway_transport::ssh_transport::TransportSender;
use tracing::{debug, info, warn};
use zerocopy::{FromBytes, IntoBytes};

use crate::cli::{CaptureBackendArg, CompressArg};

/// Maximum chunk size for frame data on the wire (64 KiB).
const CHUNK_SIZE: usize = 64 * 1024;

/// How many empty polls before yielding the thread.
const SPIN_YIELD_THRESHOLD: u32 = 64;

/// Send an anchor frame every N frames to prevent error accumulation
/// and enable resync after packet loss.
const ANCHOR_INTERVAL: u64 = 300;

/// Pack width and height into a single `u64` for `AtomicU64` sharing.
#[inline]
fn pack_resolution(width: u32, height: u32) -> u64 {
    ((width as u64) << 32) | (height as u64)
}

/// Unpack `(width, height)` from a packed `u64`. Returns `None` for 0 (= native).
#[inline]
fn unpack_resolution(packed: u64) -> Option<(u32, u32)> {
    if packed == 0 {
        return None;
    }
    let w = (packed >> 32) as u32;
    let h = (packed & 0xFFFF_FFFF) as u32;
    Some((w, h))
}

/// Nearest-neighbor downscale of an RGBA frame.
///
/// `src` is the source pixel data, `src_w × src_h` at `src_stride` bytes per row.
/// Writes into `dst` (cleared and resized internally). Returns the destination stride.
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

/// Detect and create the capture backend based on CLI args.
pub fn create_capture_backend(
    capture_arg: &CaptureBackendArg,
    output: Option<&str>,
    app_id: Option<&str>,
) -> Result<Box<dyn CaptureBackend>> {
    // Per-window capture path: ext-image-capture supports toplevel capture
    // natively; portal supports it via PortalSourceType::Window.
    if let Some(app_id) = app_id {
        return match capture_arg {
            CaptureBackendArg::WlrScreencopy => {
                anyhow::bail!(
                    "--app-id requires ext-image-capture or portal backend; \
                     wlr-screencopy does not support per-window capture"
                );
            }
            CaptureBackendArg::Auto => {
                // Try ext-image-capture first (no dialog, stable).
                // Fall back to portal per-window if available.
                match remoteway_capture::detect::detect_toplevel_backend(app_id) {
                    Ok(backend) => return Ok(backend),
                    Err(e) => {
                        tracing::info!(
                            "ext-image-capture per-window unavailable for '{app_id}': {e}"
                        );
                        try_portal_window().context(format!(
                            "failed to capture toplevel '{app_id}' \
                             (ext-image-capture and portal both unavailable)"
                        ))
                    }
                }
            }
            CaptureBackendArg::ExtImageCapture => {
                remoteway_capture::detect::detect_toplevel_backend(app_id)
                    .context(format!("failed to capture toplevel '{app_id}'"))
            }
            CaptureBackendArg::Portal => {
                try_portal_window().context("failed to create portal capture backend (window)")
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
        CaptureBackendArg::Portal => {
            try_portal_monitor().context("failed to create portal capture backend")
        }
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
        CaptureBackendArg::Auto | CaptureBackendArg::ExtImageCapture => {
            remoteway_capture::detect::detect_new_toplevel_backend(known_identifiers)
                .context("failed to detect child window")
        }
        CaptureBackendArg::Portal => {
            // Portal cannot diff against `known_identifiers` (the picker dialog
            // shows the user the full window list). The user picks the new
            // window manually.
            let _ = known_identifiers;
            try_portal_window().context("failed to create portal capture backend (window)")
        }
    }
}

/// Try portal-based capture with the given source type.
///
/// Only available when `gnome` feature is enabled.
#[cfg(feature = "gnome")]
fn try_portal_capture(
    source_type: remoteway_capture::portal::PortalSourceType,
) -> Result<Box<dyn CaptureBackend>> {
    remoteway_capture::detect::create_portal_backend(source_type)
        .context("portal capture unavailable")
}

/// Open a portal screencast session for a full monitor.
///
/// Cfg-safe wrapper: callable from code that compiles without `--features gnome`.
/// Returns a clear error at runtime when the feature isn't built in.
pub fn try_portal_monitor() -> Result<Box<dyn CaptureBackend>> {
    #[cfg(feature = "gnome")]
    {
        try_portal_capture(remoteway_capture::portal::PortalSourceType::Monitor)
    }
    #[cfg(not(feature = "gnome"))]
    {
        anyhow::bail!("portal capture requires 'gnome' feature")
    }
}

/// Open a portal screencast session for a single window (user picks via the
/// portal dialog).
pub fn try_portal_window() -> Result<Box<dyn CaptureBackend>> {
    #[cfg(feature = "gnome")]
    {
        try_portal_capture(remoteway_capture::portal::PortalSourceType::Window)
    }
    #[cfg(not(feature = "gnome"))]
    {
        anyhow::bail!("portal capture requires 'gnome' feature")
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
        CaptureBackendArg::Portal => capture_flags::PORTAL,
    };

    let comp = match compress_arg {
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
    _compress_arg: &CompressArg,
    shutdown: &AtomicBool,
    target_resolution: &AtomicU64,
) {
    let mut previous_frame: Vec<u8> = Vec::new();
    let mut is_first = true;
    let mut frame_count: u64 = 0;
    let mut empty_polls: u32 = 0;
    // When true, the next frame must use full-frame damage because the
    // compositor's damage regions are relative to a frame the client never
    // received (dropped in transport or capture ring).
    let mut need_full_damage = false;
    // Scratch buffer for downscaled frames (reused across iterations).
    let mut scaled_buf: Vec<u8> = Vec::new();
    // Track the last applied target resolution to detect changes.
    let mut last_target: u64 = 0;

    while !shutdown.load(Ordering::Acquire) {
        // Drain the capture ring to the LATEST frame. Intermediate frames
        // are stale — processing them would only add pipeline latency.
        let mut frame = match capture.try_recv() {
            Some(f) => {
                empty_polls = 0;
                f
            }
            None => {
                // If capture thread has exited (toplevel closed, session ended),
                // signal shutdown so the server terminates gracefully.
                if capture.is_finished() {
                    info!("capture thread finished, signalling shutdown");
                    shutdown.store(true, Ordering::Release);
                    break;
                }
                empty_polls += 1;
                if empty_polls >= SPIN_YIELD_THRESHOLD {
                    std::thread::yield_now();
                } else {
                    std::hint::spin_loop();
                }
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

        // Check for target resolution change — force anchor on change.
        let cur_target = target_resolution.load(Ordering::Relaxed);
        if cur_target != last_target {
            need_full_damage = true;
            // Reset delta base so the first frame at the new resolution is clean.
            previous_frame.clear();
            last_target = cur_target;
            debug!("target resolution changed, forcing anchor");
        }

        // Apply server-side downscaling if target resolution is set.
        let (frame_data, frame_w, frame_h, frame_stride): (&[u8], u32, u32, u32) =
            if let Some((tw, th)) = unpack_resolution(cur_target) {
                if tw < frame.width || th < frame.height {
                    downscale_nearest(
                        &frame.data,
                        frame.width,
                        frame.height,
                        frame.stride,
                        tw,
                        th,
                        &mut scaled_buf,
                    );
                    (&scaled_buf, tw, th, tw * 4)
                } else {
                    (&frame.data, frame.width, frame.height, frame.stride)
                }
            } else {
                (&frame.data, frame.width, frame.height, frame.stride)
            };

        // Periodic anchor frames for resync / error accumulation prevention.
        let force_anchor =
            is_first || (frame_count > 0 && frame_count.is_multiple_of(ANCHOR_INTERVAL));

        let use_full_damage = force_anchor || frame.damage.is_empty() || need_full_damage;

        let regions: Vec<DamageRect> = if use_full_damage {
            vec![DamageRect::new(0, 0, frame_w, frame_h)]
        } else if unpack_resolution(cur_target).is_some() {
            // When downscaling, compositor damage regions don't map cleanly —
            // use full-frame damage for correctness.
            vec![DamageRect::new(0, 0, frame_w, frame_h)]
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

        let compressed =
            compress_frame(frame_data, &previous_frame, frame_stride as usize, &regions);

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
            previous_frame.clear();
            previous_frame.extend_from_slice(frame_data);
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
    target_resolution: Arc<AtomicU64>,
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
            Ok(MsgType::TargetResolution) => {
                use remoteway_proto::target_resolution::TargetResolutionPayload;
                if msg.payload.len() >= TargetResolutionPayload::SIZE
                    && let Ok(tr) = TargetResolutionPayload::ref_from_bytes(
                        &msg.payload[..TargetResolutionPayload::SIZE],
                    )
                {
                    let w = tr.width;
                    let h = tr.height;
                    let packed = if w == 0 && h == 0 {
                        info!("target resolution reset to native");
                        0
                    } else {
                        info!(width = w, height = h, "target resolution set");
                        pack_resolution(w, h)
                    };
                    target_resolution.store(packed, Ordering::Relaxed);
                }
            }
            Ok(MsgType::Handshake) => {
                debug!("received client handshake");
            }
            Ok(other) => {
                debug!(?other, "ignoring unexpected message type");
            }
            Err(unknown) => {
                warn!(msg_type = unknown, "unknown message type");
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
    target_resolution: Arc<AtomicU64>,
) -> Result<JoinHandle<()>> {
    let config = ThreadConfig::new(2, 0, "compress-send");
    config
        .spawn(move || {
            compress_send_loop(
                capture,
                sender,
                &compress_arg,
                &shutdown,
                &target_resolution,
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
    fn build_handshake_portal() {
        let data = build_handshake(&CaptureBackendArg::Portal, &CompressArg::Lz4);
        use remoteway_proto::handshake::{HandshakePayload, capture_flags};
        let hs = HandshakePayload::ref_from_bytes(&data[16..24]).unwrap();
        assert_eq!({ hs.capture_flags }, capture_flags::PORTAL);
    }

    #[test]
    fn pack_unpack_round_trip() {
        assert_eq!(
            unpack_resolution(pack_resolution(1920, 1080)),
            Some((1920, 1080))
        );
        assert_eq!(
            unpack_resolution(pack_resolution(3840, 2160)),
            Some((3840, 2160))
        );
        assert_eq!(unpack_resolution(pack_resolution(1, 1)), Some((1, 1)));
        assert_eq!(
            unpack_resolution(pack_resolution(u32::MAX, u32::MAX)),
            Some((u32::MAX, u32::MAX))
        );
    }

    #[test]
    fn unpack_zero_is_none() {
        assert_eq!(unpack_resolution(0), None);
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
    fn try_portal_monitor_no_gnome_feature() {
        #[cfg(not(feature = "gnome"))]
        {
            let result = try_portal_monitor();
            assert!(result.is_err());
            let msg = format!("{}", result.err().unwrap());
            assert!(msg.contains("gnome"), "error should mention 'gnome': {msg}");
        }
    }

    #[test]
    fn try_portal_window_no_gnome_feature() {
        #[cfg(not(feature = "gnome"))]
        {
            let result = try_portal_window();
            assert!(result.is_err());
            let msg = format!("{}", result.err().unwrap());
            assert!(msg.contains("gnome"), "error should mention 'gnome': {msg}");
        }
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

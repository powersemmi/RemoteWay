//! `PipeWire` screen capture backend for portal screencast.
//!
//! Uses `ThreadLoopRc` (`pw_thread_loop`) like OBS — `set_active`
//! must be called with `pw_thread_loop_lock` held. Frames passed
//! via shared mutex + condvar.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

use pipewire::channel;

use crate::backend::{CaptureBackend, CapturedFrame, PixelFormat};
use crate::error::CaptureError;
use crate::portal::PortalSession;

pub struct PipeWireCaptureBackend {
    frame_rx: Arc<(Mutex<Option<CapturedFrame>>, Condvar)>,
    stop_tx: channel::Sender<()>,
    thread: Option<thread::JoinHandle<()>>,
}

impl PipeWireCaptureBackend {
    pub fn new(session: &PortalSession) -> Result<Self, CaptureError> {
        let pw_fd = session
            .pw_fd
            .try_clone()
            .map_err(|e| CaptureError::Protocol(format!("fd clone failed: {e}")))?;
        let pw_node_id = session.pw_node_id;
        let portal_width = session.stream_width;
        let portal_height = session.stream_height;

        let frame_rx = Arc::new((Mutex::new(None::<CapturedFrame>), Condvar::new()));
        let frame_tx = frame_rx.clone();
        let (stop_tx, stop_rx) = channel::channel::<()>();

        let thread = thread::Builder::new()
            .name("pipewire-capture".into())
            .spawn(move || {
                if let Err(e) = pipewire_thread(
                    pw_fd,
                    pw_node_id,
                    portal_width,
                    portal_height,
                    frame_tx,
                    stop_rx,
                ) {
                    tracing::error!("PipeWire capture thread failed: {e}");
                }
            })
            .map_err(|e| CaptureError::Protocol(format!("thread spawn failed: {e}")))?;

        Ok(Self {
            frame_rx,
            stop_tx,
            thread: Some(thread),
        })
    }
}

impl CaptureBackend for PipeWireCaptureBackend {
    fn next_frame(&mut self) -> Result<CapturedFrame, CaptureError> {
        let (lock, cvar) = &*self.frame_rx;
        loop {
            if let Some(ref handle) = self.thread {
                if handle.is_finished() {
                    return Err(CaptureError::SessionEnded);
                }
            } else {
                return Err(CaptureError::SessionEnded);
            }
            let guard = lock.lock().map_err(|e| {
                CaptureError::Protocol(format!("mutex poisoned in next_frame: {e}"))
            })?;
            let result = cvar
                .wait_timeout(guard, Duration::from_millis(100))
                .map_err(|e| {
                    CaptureError::Protocol(format!("condvar wait failed in next_frame: {e}"))
                })?;
            let mut guard = result.0;
            if let Some(frame) = guard.take() {
                return Ok(frame);
            }
        }
    }
    fn name(&self) -> &'static str {
        "pipewire-portal"
    }
    fn stop(&mut self) {
        if self.stop_tx.send(()).is_err() {
            tracing::debug!("pipewire stop signal: channel already closed");
        }
        if let Some(handle) = self.thread.take()
            && let Err(e) = handle.join()
        {
            tracing::warn!("pipewire capture thread panicked: {e:?}");
        }
    }
}
impl Drop for PipeWireCaptureBackend {
    fn drop(&mut self) {
        // INTENTIONAL: best-effort signal in Drop, cannot panic or propagate error
        let _ = self.stop_tx.send(());
    }
}

#[derive(Debug, Clone)]
struct VideoFormat {
    width: u32,
    height: u32,
    stride: u32,
    format: PixelFormat,
    raw_format: u32,
}

fn pipewire_thread(
    pw_fd: std::os::fd::OwnedFd,
    pw_node_id: u32,
    portal_width: u32,
    portal_height: u32,
    frame_tx: Arc<(Mutex<Option<CapturedFrame>>, Condvar)>,
    stop_rx: channel::Receiver<()>,
) -> Result<(), CaptureError> {
    use pipewire as pw;
    pw::init();

    // SAFETY: ThreadLoopRc::new calls crate::init() which is idempotent
    let thread_loop = unsafe {
        pw::thread_loop::ThreadLoopRc::new(None, None)
            .map_err(|_| CaptureError::Protocol("ThreadLoop creation failed".into()))?
    };

    let context = pw::context::ContextRc::new(&thread_loop, None)
        .map_err(|_| CaptureError::Protocol("Context creation failed".into()))?;
    let core = context
        .connect_fd_rc(pw_fd, None)
        .map_err(|_| CaptureError::Protocol("connect_fd failed".into()))?;

    tracing::info!(
        pw_node_id,
        portal_width,
        portal_height,
        "connecting PipeWire stream"
    );

    // Stop flag — set when stop_rx fires
    let stopped = Arc::new(AtomicBool::new(false));
    let stopped_cb = stopped.clone();

    // Attach stop handler
    let tl_stop = thread_loop.clone();
    let _stop_handle = stop_rx.attach(thread_loop.loop_(), move |_| {
        stopped_cb.store(true, Ordering::Release);
        tl_stop.stop();
    });

    // node.target + PW_ID_ANY
    let mut props = pw::properties::PropertiesBox::new();
    props.insert("media.type", "Video");
    props.insert("media.category", "Capture");
    props.insert("media.role", "Screen");
    props.insert("node.target", pw_node_id.to_string());

    let stream = pw::stream::StreamBox::new(&core, "remoteway-capture", props)
        .map_err(|_| CaptureError::Protocol("Stream creation failed".into()))?;

    let format: Arc<Mutex<Option<VideoFormat>>> = Arc::new(Mutex::new(None));
    let fmt_set = Arc::clone(&format);
    let fmt_get = Arc::clone(&format);
    let frame_tx_cb = frame_tx.clone();
    let frame_count = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let frame_count_cb = frame_count.clone();
    let needs_reconnect = Arc::new(AtomicBool::new(false));
    let needs_reconnect_cb = needs_reconnect.clone();

    let _listener = stream
        .add_local_listener_with_user_data(())
        .state_changed(move |_s, _d, old, new| {
            tracing::info!("PW stream: {old:?} -> {new:?}");
            if let pw::stream::StreamState::Error(ref msg) = new {
                tracing::error!(msg, "PW stream error");
            }
            if new == pw::stream::StreamState::Paused && old == pw::stream::StreamState::Streaming {
                needs_reconnect_cb.store(true, Ordering::Release);
            }
        })
        .param_changed(move |stream, _d, id, pod| {
            if id != pw::spa::param::ParamType::Format.as_raw() {
                return;
            }
            let Some(pod) = pod else { return };
            let fmt = match parse_video_format(pod) {
                Ok(Some(f)) => f,
                Ok(None) => return,
                Err(e) => {
                    tracing::error!(?e, "parse_video_format failed");
                    return;
                }
            };
            tracing::info!(
                width = fmt.width,
                height = fmt.height,
                stride = fmt.stride,
                "format negotiated"
            );
            let stride = fmt.stride;
            let height = fmt.height;
            if let Ok(mut g) = fmt_set.lock() {
                *g = Some(fmt);
            }
            // Critical: complete negotiation by pushing SPA_PARAM_Buffers + SPA_PARAM_Meta.
            // Without this KWin/portal producer stops after 1 frame (buffer pool never finalized).
            let response = match build_param_response_pods(stride, height) {
                Ok(r) => r,
                Err(e) => {
                    tracing::error!(?e, "build_param_response_pods failed");
                    return;
                }
            };
            let mut refs: Vec<&pw::spa::pod::Pod> = response
                .iter()
                .filter_map(|b| pw::spa::pod::Pod::from_bytes(b))
                .collect();
            match stream.update_params(&mut refs) {
                Ok(()) => tracing::info!(n = refs.len(), "update_params: buffers + meta pushed"),
                Err(e) => tracing::error!(?e, "update_params failed"),
            }
        })
        .process(move |_s, _d| {
            let fmt = match fmt_get.lock() {
                Ok(g) => g.clone(),
                Err(_) => return,
            };
            let Some(fmt) = fmt else { return };

            let mut buffer = match _s.dequeue_buffer() {
                Some(b) => b,
                None => {
                    tracing::trace!("process: no buffer to dequeue");
                    return;
                }
            };
            let frame = {
                let chunk = match buffer.datas_mut().first_mut() {
                    Some(c) => c,
                    None => return,
                };
                let slice = match chunk.data() {
                    Some(s) => s,
                    None => {
                        tracing::warn!("process: chunk.data() returned None (unmappable buffer)");
                        return;
                    }
                };
                let expected = (fmt.stride * fmt.height) as usize;
                if slice.len() < expected {
                    tracing::warn!(
                        got = slice.len(),
                        expected,
                        "process: short buffer, skipping"
                    );
                    return;
                }
                let src = &slice[..expected];
                let (data, format, stride) =
                    if fmt.raw_format == pipewire::spa::sys::SPA_VIDEO_FORMAT_YUY2 {
                        let bgra = yuy2_to_bgra(src, fmt.width, fmt.height);
                        (bgra, PixelFormat::Argb8888, fmt.width * 4)
                    } else {
                        (src.to_vec(), fmt.format, fmt.stride)
                    };
                CapturedFrame {
                    data,
                    damage: vec![],
                    format,
                    width: fmt.width,
                    height: fmt.height,
                    stride,
                    timestamp_ns: 0,
                }
            };
            drop(buffer);
            let n = frame_count_cb.fetch_add(1, Ordering::Relaxed) + 1;
            if n == 1 || n.is_multiple_of(60) {
                tracing::info!(n, "process: frame delivered");
            }
            let (lk, cv) = &*frame_tx_cb;
            if let Ok(mut p) = lk.lock() {
                *p = Some(frame);
                cv.notify_one();
            }
        })
        .register()
        .map_err(|_| CaptureError::Protocol("listener registration failed".into()))?;

    // Format pods
    let fpods = build_format_params(portal_width, portal_height)?;
    let mut refs: Vec<&pw::spa::pod::Pod> = fpods
        .iter()
        .filter_map(|b| pw::spa::pod::Pod::from_bytes(b))
        .collect();

    // Connect WITH lock held
    {
        let _g = thread_loop.lock();
        stream
            .connect(
                pw::spa::utils::Direction::Input,
                None,
                pw::stream::StreamFlags::AUTOCONNECT
                    | pw::stream::StreamFlags::MAP_BUFFERS
                    | pw::stream::StreamFlags::DRIVER
                    | pw::stream::StreamFlags::RT_PROCESS,
                &mut refs,
            )
            .map_err(|_| CaptureError::Protocol("connect failed".into()))?;
    }

    thread_loop.start();
    tracing::info!("PipeWire stream connected, thread loop running");

    // Main loop: check needs_reconnect every 100ms, exit on stop
    let (lk, cv) = &*frame_tx;
    let mut guard = lk
        .lock()
        .map_err(|e| CaptureError::Protocol(format!("mutex poisoned in pipewire_thread: {e}")))?;
    while !stopped.load(Ordering::Acquire) {
        if needs_reconnect.swap(false, Ordering::AcqRel) {
            tracing::info!("set_active(false/true) with lock");
            let _g = thread_loop.lock();
            if let Err(e) = stream.set_active(false) {
                tracing::warn!("pipewire set_active(false) failed: {e:?}");
            }
            if let Err(e) = stream.set_active(true) {
                tracing::warn!("pipewire set_active(true) failed: {e:?}");
            }
        }
        guard = cv
            .wait_timeout(guard, Duration::from_millis(100))
            .map_err(|e| {
                CaptureError::Protocol(format!("condvar wait failed in pipewire_thread: {e}"))
            })?
            .0;
    }
    drop(guard);

    thread_loop.stop();
    tracing::info!("PipeWire capture stopped");
    Ok(())
}

// Helper functions

fn build_format_params(width: u32, height: u32) -> Result<Vec<Vec<u8>>, CaptureError> {
    use pipewire::spa::sys;
    let formats = [
        sys::SPA_VIDEO_FORMAT_BGRA,
        sys::SPA_VIDEO_FORMAT_BGRx,
        sys::SPA_VIDEO_FORMAT_RGBA,
        sys::SPA_VIDEO_FORMAT_RGBx,
    ];
    let pod = format_enum_pod(width, height, &formats)?;
    Ok(vec![pod])
}

fn format_enum_pod(width: u32, height: u32, formats: &[u32]) -> Result<Vec<u8>, CaptureError> {
    use pipewire::spa::{param, pod, utils};
    let mut properties = vec![
        pod::Property {
            key: param::format::FormatProperties::MediaType.as_raw(),
            flags: pod::PropertyFlags::empty(),
            value: pod::Value::Id(utils::Id(param::format::MediaType::Video.as_raw())),
        },
        pod::Property {
            key: param::format::FormatProperties::MediaSubtype.as_raw(),
            flags: pod::PropertyFlags::empty(),
            value: pod::Value::Id(utils::Id(param::format::MediaSubtype::Raw.as_raw())),
        },
        pod::Property {
            key: param::format::FormatProperties::VideoSize.as_raw(),
            flags: pod::PropertyFlags::empty(),
            value: pod::Value::Choice(pod::ChoiceValue::Rectangle(utils::Choice(
                utils::ChoiceFlags::empty(),
                utils::ChoiceEnum::Range {
                    default: utils::Rectangle { width, height },
                    min: utils::Rectangle {
                        width: 1,
                        height: 1,
                    },
                    max: utils::Rectangle {
                        width: 8192,
                        height: 8192,
                    },
                },
            ))),
        },
        pod::Property {
            key: param::format::FormatProperties::VideoFramerate.as_raw(),
            flags: pod::PropertyFlags::empty(),
            value: pod::Value::Choice(pod::ChoiceValue::Fraction(utils::Choice(
                utils::ChoiceFlags::empty(),
                utils::ChoiceEnum::Range {
                    default: utils::Fraction { num: 30, denom: 1 },
                    min: utils::Fraction { num: 0, denom: 1 },
                    max: utils::Fraction {
                        num: 1000,
                        denom: 1,
                    },
                },
            ))),
        },
    ];
    let ids: Vec<utils::Id> = formats.iter().map(|f| utils::Id(*f)).collect();
    properties.push(pod::Property {
        key: param::format::FormatProperties::VideoFormat.as_raw(),
        flags: pod::PropertyFlags::empty(),
        value: pod::Value::Choice(pod::ChoiceValue::Id(utils::Choice(
            utils::ChoiceFlags::empty(),
            utils::ChoiceEnum::Enum {
                default: ids[0],
                alternatives: ids,
            },
        ))),
    });
    let obj = pod::Object {
        type_: utils::SpaTypes::ObjectParamFormat.as_raw(),
        id: param::ParamType::EnumFormat.as_raw(),
        properties,
    };
    let mut cursor = std::io::Cursor::new(Vec::new());
    // INTENTIONAL: serialize writes into cursor; result tuple not needed
    let _ = pod::serialize::PodSerializer::serialize(&mut cursor, &pod::Value::Object(obj))
        .map_err(|_| CaptureError::Protocol("pod serialization failed".into()))?;
    Ok(cursor.into_inner())
}

/// Build the parameters the consumer must push back via `pw_stream_update_params`
/// after format negotiation. Without `SPA_PARAM_Buffers` and `SPA_PARAM_Meta(Header)`
/// `KWin`'s screencast producer stops after the first buffer (pool never finalized).
///
/// Mirrors what kpipewire / OBS / lamco-pipewire do.
fn build_param_response_pods(stride: u32, height: u32) -> Result<Vec<Vec<u8>>, CaptureError> {
    let out = vec![
        build_buffers_pod(stride, height)?,
        build_meta_pod(pipewire::spa::sys::SPA_META_Header, 32)?,
        // Optional but commonly requested:
        build_meta_pod(pipewire::spa::sys::SPA_META_VideoCrop, 16)?,
        build_meta_pod(pipewire::spa::sys::SPA_META_VideoTransform, 4)?,
        build_meta_pod(pipewire::spa::sys::SPA_META_VideoDamage, 16 * 16)?,
    ];
    Ok(out)
}

fn build_buffers_pod(stride: u32, height: u32) -> Result<Vec<u8>, CaptureError> {
    use pipewire::spa::{param, pod, sys, utils};
    let size = (stride.saturating_mul(height)) as i32;
    let mem_mask = ((1u32 << sys::SPA_DATA_MemPtr) | (1u32 << sys::SPA_DATA_MemFd)) as i32;
    let properties = vec![
        pod::Property {
            key: sys::SPA_PARAM_BUFFERS_buffers,
            flags: pod::PropertyFlags::empty(),
            value: pod::Value::Choice(pod::ChoiceValue::Int(utils::Choice(
                utils::ChoiceFlags::empty(),
                utils::ChoiceEnum::Range {
                    default: 8,
                    min: 2,
                    max: 16,
                },
            ))),
        },
        pod::Property {
            key: sys::SPA_PARAM_BUFFERS_blocks,
            flags: pod::PropertyFlags::empty(),
            value: pod::Value::Int(1),
        },
        pod::Property {
            key: sys::SPA_PARAM_BUFFERS_size,
            flags: pod::PropertyFlags::empty(),
            value: pod::Value::Choice(pod::ChoiceValue::Int(utils::Choice(
                utils::ChoiceFlags::empty(),
                utils::ChoiceEnum::Range {
                    default: size,
                    min: size,
                    max: i32::MAX,
                },
            ))),
        },
        pod::Property {
            key: sys::SPA_PARAM_BUFFERS_stride,
            flags: pod::PropertyFlags::empty(),
            value: pod::Value::Int(stride as i32),
        },
        pod::Property {
            key: sys::SPA_PARAM_BUFFERS_align,
            flags: pod::PropertyFlags::empty(),
            value: pod::Value::Int(16),
        },
        pod::Property {
            key: sys::SPA_PARAM_BUFFERS_dataType,
            flags: pod::PropertyFlags::empty(),
            value: pod::Value::Choice(pod::ChoiceValue::Int(utils::Choice(
                utils::ChoiceFlags::empty(),
                utils::ChoiceEnum::Flags {
                    default: mem_mask,
                    flags: vec![mem_mask],
                },
            ))),
        },
    ];
    let obj = pod::Object {
        type_: utils::SpaTypes::ObjectParamBuffers.as_raw(),
        id: param::ParamType::Buffers.as_raw(),
        properties,
    };
    let mut cursor = std::io::Cursor::new(Vec::new());
    // INTENTIONAL: serialize writes into cursor; result tuple not needed
    let _ = pod::serialize::PodSerializer::serialize(&mut cursor, &pod::Value::Object(obj))
        .map_err(|_| CaptureError::Protocol("pod serialization failed".into()))?;
    Ok(cursor.into_inner())
}

fn build_meta_pod(meta_type: u32, size: u32) -> Result<Vec<u8>, CaptureError> {
    use pipewire::spa::{param, pod, sys, utils};
    let properties = vec![
        pod::Property {
            key: sys::SPA_PARAM_META_type,
            flags: pod::PropertyFlags::empty(),
            value: pod::Value::Id(utils::Id(meta_type)),
        },
        pod::Property {
            key: sys::SPA_PARAM_META_size,
            flags: pod::PropertyFlags::empty(),
            value: pod::Value::Int(size as i32),
        },
    ];
    let obj = pod::Object {
        type_: utils::SpaTypes::ObjectParamMeta.as_raw(),
        id: param::ParamType::Meta.as_raw(),
        properties,
    };
    let mut cursor = std::io::Cursor::new(Vec::new());
    // INTENTIONAL: serialize writes into cursor; result tuple not needed
    let _ = pod::serialize::PodSerializer::serialize(&mut cursor, &pod::Value::Object(obj))
        .map_err(|_| CaptureError::Protocol("pod serialization failed".into()))?;
    Ok(cursor.into_inner())
}

fn parse_video_format(pod: &pipewire::spa::pod::Pod) -> Result<Option<VideoFormat>, CaptureError> {
    use pipewire::spa::param::video::{VideoFormat as PwVideoFormat, VideoInfoRaw};
    let mut info = VideoInfoRaw::default();
    // INTENTIONAL: parse populates `info` in-place; SpaSuccess value not needed
    let _ = info
        .parse(pod)
        .map_err(|_| CaptureError::Protocol("pod parse failed".into()))?;
    let pw_fmt = info.format();
    if pw_fmt == PwVideoFormat::Unknown {
        return Ok(None);
    }
    let (format, bpp) = match pw_fmt {
        PwVideoFormat::BGRA => (PixelFormat::Argb8888, 4),
        PwVideoFormat::BGRx => (PixelFormat::Xrgb8888, 4),
        PwVideoFormat::RGBA => (PixelFormat::Abgr8888, 4),
        PwVideoFormat::RGBx => (PixelFormat::Xbgr8888, 4),
        PwVideoFormat::ARGB => (PixelFormat::Argb8888, 4),
        PwVideoFormat::ABGR => (PixelFormat::Abgr8888, 4),
        PwVideoFormat::xRGB => (PixelFormat::Xrgb8888, 4),
        PwVideoFormat::xBGR => (PixelFormat::Xbgr8888, 4),
        PwVideoFormat::YUY2 => (PixelFormat::Argb8888, 2),
        _ => {
            tracing::warn!(?pw_fmt, "unsupported format");
            return Ok(None);
        }
    };
    let size = info.size();
    Ok(Some(VideoFormat {
        width: size.width,
        height: size.height,
        stride: size.width * bpp,
        format,
        raw_format: pw_fmt.as_raw(),
    }))
}

fn yuy2_to_bgra(src: &[u8], width: u32, height: u32) -> Vec<u8> {
    let pixel_count = (width * height) as usize;
    let mut dst = vec![0u8; pixel_count * 4];
    for row in 0..height as usize {
        let srow = &src[row * (width as usize * 2)..];
        let drow = &mut dst[row * (width as usize * 4)..];
        for x in 0..(width as usize / 2) {
            let b = x * 4;
            let y0 = srow[b] as i32;
            let u = srow[b + 1] as i32 - 128;
            let y1 = srow[b + 2] as i32;
            let v = srow[b + 3] as i32 - 128;
            let db = x * 8;
            drow[db] = clamp_u8(y0 + ((u * 516 + 128) >> 8));
            drow[db + 1] = clamp_u8(y0 - ((u * 100 + v * 208 - 128) >> 8));
            drow[db + 2] = clamp_u8(y0 + ((v * 409 + 128) >> 8));
            drow[db + 3] = 255;
            drow[db + 4] = clamp_u8(y1 + ((u * 516 + 128) >> 8));
            drow[db + 5] = clamp_u8(y1 - ((u * 100 + v * 208 - 128) >> 8));
            drow[db + 6] = clamp_u8(y1 + ((v * 409 + 128) >> 8));
            drow[db + 7] = 255;
        }
    }
    dst
}

#[inline]
fn clamp_u8(val: i32) -> u8 {
    val.clamp(0, 255) as u8
}

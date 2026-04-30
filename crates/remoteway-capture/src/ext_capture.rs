use tracing::warn;
use wayland_client::protocol::{wl_buffer, wl_output, wl_registry, wl_shm, wl_shm_pool};
use wayland_client::{Connection, Dispatch, QueueHandle, WEnum};

use remoteway_compress::delta::DamageRect;

use crate::backend::{CaptureBackend, CapturedFrame, PixelFormat};
use crate::error::CaptureError;
use crate::protocols::ext_foreign_toplevel_list_v1::client::{
    ext_foreign_toplevel_handle_v1, ext_foreign_toplevel_list_v1,
};
use crate::protocols::ext_image_capture_source_v1::client::{
    ext_foreign_toplevel_image_capture_source_manager_v1, ext_image_capture_source_v1,
    ext_output_image_capture_source_manager_v1,
};
use crate::protocols::ext_image_copy_capture_v1::client::{
    ext_image_copy_capture_frame_v1, ext_image_copy_capture_manager_v1,
    ext_image_copy_capture_session_v1,
};
use crate::shm::ShmBufferPool;

/// ext-image-capture backend using the standard Wayland capture protocol.
///
/// Supports per-output and per-toplevel capture. Preferred over wlr-screencopy
/// when available (KDE 6.2+, wlroots with ext-image-capture support).
pub struct ExtImageCaptureBackend {
    _conn: Connection,
    state: ExtCaptureState,
    event_queue: wayland_client::EventQueue<ExtCaptureState>,
}

impl std::fmt::Debug for ExtImageCaptureBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExtImageCaptureBackend")
            .finish_non_exhaustive()
    }
}

/// Source type for capture.
pub enum CaptureSource {
    /// Capture an output by name. `None` = first available.
    Output(Option<String>),
    /// Capture a toplevel by app_id.
    Toplevel(String),
    /// Capture the first toplevel whose identifier is NOT in the known set.
    /// Used for auto-detecting a newly spawned child's window.
    NewToplevel { known_identifiers: Vec<String> },
}

/// Public information about a discovered toplevel window.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ToplevelInfo {
    pub app_id: String,
    pub title: String,
    /// Compositor-specific unique identifier (stable for the toplevel's lifetime).
    pub identifier: String,
}

struct DiscoveredOutput {
    wl_output: wl_output::WlOutput,
    /// Compositor name (e.g. "HDMI-A-1"), filled from wl_output.name event.
    name: String,
    /// Wayland global name for matching wl_output events to this entry.
    global_name: u32,
}

struct ExtCaptureState {
    // Globals.
    shm: Option<wl_shm::WlShm>,
    output_source_manager:
        Option<ext_output_image_capture_source_manager_v1::ExtOutputImageCaptureSourceManagerV1>,
    toplevel_source_manager: Option<
        ext_foreign_toplevel_image_capture_source_manager_v1::ExtForeignToplevelImageCaptureSourceManagerV1,
    >,
    copy_capture_manager:
        Option<ext_image_copy_capture_manager_v1::ExtImageCopyCaptureManagerV1>,
    toplevel_list: Option<ext_foreign_toplevel_list_v1::ExtForeignToplevelListV1>,
    // Discovered objects.
    outputs: Vec<DiscoveredOutput>,
    toplevels: Vec<DiscoveredToplevel>,
    // Session.
    source: Option<ext_image_capture_source_v1::ExtImageCaptureSourceV1>,
    session: Option<ext_image_copy_capture_session_v1::ExtImageCopyCaptureSessionV1>,
    shm_pool: Option<ShmBufferPool>,
    // Session params (from session events).
    buffer_width: u32,
    buffer_height: u32,
    shm_format: Option<wl_shm::Format>,
    session_ready: bool,
    // Per-frame state.
    frame_ready: bool,
    frame_failed: bool,
    damage_rects: Vec<DamageRect>,
    timestamp_ns: u64,
    stopped: bool,
}

struct DiscoveredToplevel {
    handle: ext_foreign_toplevel_handle_v1::ExtForeignToplevelHandleV1,
    app_id: String,
    title: String,
    identifier: String,
}

impl ExtImageCaptureBackend {
    /// Create a new ext-image-capture backend.
    ///
    /// Returns `Err(NoBackend)` if the compositor doesn't support the protocol.
    pub fn new(source: CaptureSource) -> Result<Self, CaptureError> {
        let conn = Connection::connect_to_env()?;
        let display = conn.display();
        let mut event_queue = conn.new_event_queue::<ExtCaptureState>();
        let qh = event_queue.handle();

        let mut state = ExtCaptureState {
            shm: None,
            output_source_manager: None,
            toplevel_source_manager: None,
            copy_capture_manager: None,
            toplevel_list: None,
            outputs: Vec::new(),
            toplevels: Vec::new(),
            source: None,
            session: None,
            shm_pool: None,
            buffer_width: 0,
            buffer_height: 0,
            shm_format: None,
            session_ready: false,
            frame_ready: false,
            frame_failed: false,
            damage_rects: Vec::new(),
            timestamp_ns: 0,
            stopped: false,
        };

        display.get_registry(&qh, ());
        event_queue.roundtrip(&mut state)?;

        // Verify required globals.
        if state.copy_capture_manager.is_none() {
            warn!(
                has_toplevel_source_mgr = state.toplevel_source_manager.is_some(),
                has_toplevel_list = state.toplevel_list.is_some(),
                has_output_source_mgr = state.output_source_manager.is_some(),
                "ext_image_copy_capture_manager_v1 not available — \
                 compositor does not support ext-image-capture protocol"
            );
            return Err(CaptureError::NoBackend);
        }
        if state.shm.is_none() {
            return Err(CaptureError::CaptureFailed("wl_shm not available".into()));
        }

        // Second round-trip: get output names and toplevel info.
        event_queue.roundtrip(&mut state)?;

        // Create capture source based on requested type.
        match source {
            CaptureSource::Output(ref name) => {
                let manager = state
                    .output_source_manager
                    .as_ref()
                    .ok_or(CaptureError::NoBackend)?;
                let output = if let Some(name) = name {
                    &state
                        .outputs
                        .iter()
                        .find(|o| o.name == *name)
                        .ok_or_else(|| CaptureError::OutputNotFound(name.clone()))?
                        .wl_output
                } else {
                    &state
                        .outputs
                        .first()
                        .ok_or(CaptureError::NoOutputs)?
                        .wl_output
                };
                let src = manager.create_source(output, &qh, ());
                state.source = Some(src);
            }
            CaptureSource::Toplevel(ref app_id) => {
                let manager = state.toplevel_source_manager.as_ref().ok_or_else(|| {
                    warn!(
                        has_copy_capture = state.copy_capture_manager.is_some(),
                        has_toplevel_list = state.toplevel_list.is_some(),
                        "ext_foreign_toplevel_image_capture_source_manager_v1 not available — \
                         compositor does not support per-window capture"
                    );
                    CaptureError::NoBackend
                })?;
                let toplevel = state
                    .toplevels
                    .iter()
                    .find(|t| t.app_id == *app_id)
                    .ok_or_else(|| {
                        let available: Vec<&str> =
                            state.toplevels.iter().map(|t| t.app_id.as_str()).collect();
                        warn!(
                            target_app_id = app_id.as_str(),
                            ?available,
                            "toplevel not found among {} discovered windows",
                            state.toplevels.len()
                        );
                        CaptureError::CaptureFailed(format!(
                            "toplevel '{}' not found; available: {:?}",
                            app_id, available
                        ))
                    })?;
                let src = manager.create_source(&toplevel.handle, &qh, ());
                state.source = Some(src);
            }
            CaptureSource::NewToplevel {
                ref known_identifiers,
            } => {
                let manager = state.toplevel_source_manager.as_ref().ok_or_else(|| {
                    warn!(
                        "ext_foreign_toplevel_image_capture_source_manager_v1 not available — \
                         compositor does not support per-window capture"
                    );
                    CaptureError::NoBackend
                })?;

                // Strategy to find the "new" toplevel:
                // 1. Prefer a toplevel with non-empty identifier not in known_identifiers.
                // 2. When known_identifiers is empty (snapshot found 0 windows),
                //    treat ANY toplevel as new — covers compositors that skip
                //    the Identifier event (e.g. Hyprland on some versions).
                let toplevel = state
                    .toplevels
                    .iter()
                    .find(|t| {
                        !t.identifier.is_empty() && !known_identifiers.contains(&t.identifier)
                    })
                    .or_else(|| {
                        if known_identifiers.is_empty() {
                            // No known toplevels exist: the first toplevel
                            // (even with empty identifier) must be the new one.
                            state.toplevels.first()
                        } else {
                            None
                        }
                    })
                    .ok_or_else(|| {
                        let visible: Vec<(&str, &str)> = state
                            .toplevels
                            .iter()
                            .map(|t| (t.app_id.as_str(), t.identifier.as_str()))
                            .collect();
                        warn!(
                            known_count = known_identifiers.len(),
                            current_count = state.toplevels.len(),
                            ?visible,
                            "no new toplevel found"
                        );
                        CaptureError::CaptureFailed("no new toplevel found".into())
                    })?;
                let src = manager.create_source(&toplevel.handle, &qh, ());
                state.source = Some(src);
            }
        }

        // Create capture session.
        let capture_manager = state.copy_capture_manager.as_ref().unwrap();
        let source = state.source.as_ref().unwrap();
        let session = capture_manager.create_session(
            source,
            ext_image_copy_capture_manager_v1::Options::empty(),
            &qh,
            (),
        );
        state.session = Some(session);

        // Round-trip to get session parameters (buffer_size, shm_format, done).
        event_queue.roundtrip(&mut state)?;

        if !state.session_ready {
            return Err(CaptureError::CaptureFailed(
                "session setup incomplete".into(),
            ));
        }

        // Create SHM pool now that we know dimensions.
        let format = state.shm_format.ok_or(CaptureError::CaptureFailed(
            "no supported SHM format".into(),
        ))?;
        let stride = state.buffer_width * 4;
        let pool = ShmBufferPool::new(
            state.shm.as_ref().unwrap(),
            state.buffer_width,
            state.buffer_height,
            stride,
            format,
            &qh,
        )?;
        state.shm_pool = Some(pool);

        Ok(Self {
            _conn: conn,
            state,
            event_queue,
        })
    }

    /// Check if ext-image-capture is available on the given connection.
    pub fn is_available(conn: &Connection) -> bool {
        let mut eq = conn.new_event_queue::<ExtCaptureProbe>();
        let qh = eq.handle();
        let mut probe = ExtCaptureProbe { found: false };
        conn.display().get_registry(&qh, ());
        let _ = eq.roundtrip(&mut probe);
        probe.found
    }
}

impl CaptureBackend for ExtImageCaptureBackend {
    fn next_frame(&mut self) -> Result<CapturedFrame, CaptureError> {
        if self.state.stopped {
            return Err(CaptureError::SessionEnded);
        }

        let qh = self.event_queue.handle();

        // Reset per-frame state.
        self.state.frame_ready = false;
        self.state.frame_failed = false;
        self.state.damage_rects.clear();

        // Create a frame request.
        let session = self.state.session.as_ref().ok_or(CaptureError::NoBackend)?;
        let frame = session.create_frame(&qh, ());

        // Attach buffer.
        let pool = self
            .state
            .shm_pool
            .as_ref()
            .ok_or(CaptureError::CaptureFailed("no SHM pool".into()))?;
        frame.attach_buffer(pool.active_buffer());

        // Hint full damage (we want the whole frame).
        frame.damage_buffer(
            0,
            0,
            self.state.buffer_width as i32,
            self.state.buffer_height as i32,
        );

        // Trigger capture.
        frame.capture();

        // Dispatch until ready or failed.
        while !self.state.frame_ready && !self.state.frame_failed {
            self.event_queue.blocking_dispatch(&mut self.state)?;
        }

        // Destroy the frame object — otherwise the compositor rejects
        // a subsequent create_frame as "duplicate frame".
        frame.destroy();

        if self.state.frame_failed {
            return Err(CaptureError::CaptureFailed(
                "compositor reported capture failure".into(),
            ));
        }

        // If no damage reported, treat entire frame as damaged.
        if self.state.damage_rects.is_empty() {
            self.state.damage_rects.push(DamageRect::new(
                0,
                0,
                self.state.buffer_width,
                self.state.buffer_height,
            ));
        }

        let stride = self.state.buffer_width * 4;
        let format_raw = self.state.shm_format.map(|f| f as u32).unwrap_or(0);
        let format = PixelFormat::from_wl_shm(format_raw)
            .ok_or(CaptureError::UnsupportedFormat(format_raw))?;

        // SAFETY: frame_ready == true means the compositor has finished writing.
        let data = unsafe { self.state.shm_pool.as_ref().unwrap().active_data().to_vec() };

        // Swap double buffer.
        if let Some(ref mut pool) = self.state.shm_pool {
            pool.swap();
        }

        Ok(CapturedFrame {
            data,
            damage: std::mem::take(&mut self.state.damage_rects),
            format,
            width: self.state.buffer_width,
            height: self.state.buffer_height,
            stride,
            timestamp_ns: self.state.timestamp_ns,
        })
    }

    fn name(&self) -> &'static str {
        "ext-image-capture"
    }

    fn stop(&mut self) {
        self.state.stopped = true;
    }
}

// --- Dispatch implementations ---

impl Dispatch<wl_registry::WlRegistry, ()> for ExtCaptureState {
    fn event(
        state: &mut Self,
        registry: &wl_registry::WlRegistry,
        event: wl_registry::Event,
        _: &(),
        _conn: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if let wl_registry::Event::Global {
            name,
            interface,
            version,
        } = event
        {
            match interface.as_str() {
                "wl_shm" => {
                    state.shm = Some(registry.bind(name, version.min(1), qh, ()));
                }
                "wl_output" => {
                    let output: wl_output::WlOutput = registry.bind(name, version.min(4), qh, name);
                    state.outputs.push(DiscoveredOutput {
                        wl_output: output,
                        name: String::new(),
                        global_name: name,
                    });
                }
                "ext_output_image_capture_source_manager_v1" => {
                    state.output_source_manager = Some(registry.bind(name, version.min(1), qh, ()));
                }
                "ext_foreign_toplevel_image_capture_source_manager_v1" => {
                    state.toplevel_source_manager =
                        Some(registry.bind(name, version.min(1), qh, ()));
                }
                "ext_image_copy_capture_manager_v1" => {
                    state.copy_capture_manager = Some(registry.bind(name, version.min(1), qh, ()));
                }
                "ext_foreign_toplevel_list_v1" => {
                    state.toplevel_list = Some(registry.bind(name, version.min(1), qh, ()));
                }
                _ => {}
            }
        }
    }
}

impl Dispatch<wl_shm::WlShm, ()> for ExtCaptureState {
    fn event(
        _: &mut Self,
        _: &wl_shm::WlShm,
        _: wl_shm::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<wl_shm_pool::WlShmPool, ()> for ExtCaptureState {
    fn event(
        _: &mut Self,
        _: &wl_shm_pool::WlShmPool,
        _: wl_shm_pool::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<wl_buffer::WlBuffer, usize> for ExtCaptureState {
    fn event(
        _: &mut Self,
        _: &wl_buffer::WlBuffer,
        _: wl_buffer::Event,
        _: &usize,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<wl_output::WlOutput, u32> for ExtCaptureState {
    fn event(
        state: &mut Self,
        _proxy: &wl_output::WlOutput,
        event: wl_output::Event,
        global_name: &u32,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        if let wl_output::Event::Name { name } = event
            && let Some(out) = state
                .outputs
                .iter_mut()
                .find(|o| o.global_name == *global_name)
        {
            out.name = name;
        }
    }
}

impl Dispatch<ext_output_image_capture_source_manager_v1::ExtOutputImageCaptureSourceManagerV1, ()>
    for ExtCaptureState
{
    fn event(
        _: &mut Self,
        _: &ext_output_image_capture_source_manager_v1::ExtOutputImageCaptureSourceManagerV1,
        _: ext_output_image_capture_source_manager_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl
    Dispatch<
        ext_foreign_toplevel_image_capture_source_manager_v1::ExtForeignToplevelImageCaptureSourceManagerV1,
        (),
    > for ExtCaptureState
{
    fn event(
        _: &mut Self,
        _: &ext_foreign_toplevel_image_capture_source_manager_v1::ExtForeignToplevelImageCaptureSourceManagerV1,
        _: ext_foreign_toplevel_image_capture_source_manager_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<ext_image_capture_source_v1::ExtImageCaptureSourceV1, ()> for ExtCaptureState {
    fn event(
        _: &mut Self,
        _: &ext_image_capture_source_v1::ExtImageCaptureSourceV1,
        _: ext_image_capture_source_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<ext_image_copy_capture_manager_v1::ExtImageCopyCaptureManagerV1, ()>
    for ExtCaptureState
{
    fn event(
        _: &mut Self,
        _: &ext_image_copy_capture_manager_v1::ExtImageCopyCaptureManagerV1,
        _: ext_image_copy_capture_manager_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<ext_image_copy_capture_session_v1::ExtImageCopyCaptureSessionV1, ()>
    for ExtCaptureState
{
    fn event(
        state: &mut Self,
        _proxy: &ext_image_copy_capture_session_v1::ExtImageCopyCaptureSessionV1,
        event: ext_image_copy_capture_session_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            ext_image_copy_capture_session_v1::Event::BufferSize { width, height } => {
                state.buffer_width = width;
                state.buffer_height = height;
            }
            ext_image_copy_capture_session_v1::Event::ShmFormat { format } => {
                // Prefer XRGB8888 (format 1), but accept first offered.
                let is_xrgb = matches!(format, WEnum::Value(wl_shm::Format::Xrgb8888));
                if (state.shm_format.is_none() || is_xrgb)
                    && let WEnum::Value(f) = format
                {
                    state.shm_format = Some(f);
                }
            }
            ext_image_copy_capture_session_v1::Event::Done => {
                state.session_ready = true;
            }
            ext_image_copy_capture_session_v1::Event::Stopped => {
                state.stopped = true;
            }
            _ => {}
        }
    }
}

impl Dispatch<ext_image_copy_capture_frame_v1::ExtImageCopyCaptureFrameV1, ()> for ExtCaptureState {
    fn event(
        state: &mut Self,
        _proxy: &ext_image_copy_capture_frame_v1::ExtImageCopyCaptureFrameV1,
        event: ext_image_copy_capture_frame_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            ext_image_copy_capture_frame_v1::Event::Damage {
                x,
                y,
                width,
                height,
            } => {
                state.damage_rects.push(DamageRect::new(
                    x as u32,
                    y as u32,
                    width as u32,
                    height as u32,
                ));
            }
            ext_image_copy_capture_frame_v1::Event::PresentationTime {
                tv_sec_hi,
                tv_sec_lo,
                tv_nsec,
            } => {
                state.timestamp_ns =
                    ((tv_sec_hi as u64) << 32 | tv_sec_lo as u64) * 1_000_000_000 + tv_nsec as u64;
            }
            ext_image_copy_capture_frame_v1::Event::Ready => {
                state.frame_ready = true;
            }
            ext_image_copy_capture_frame_v1::Event::Failed { .. } => {
                state.frame_failed = true;
            }
            _ => {}
        }
    }
}

impl Dispatch<ext_foreign_toplevel_list_v1::ExtForeignToplevelListV1, ()> for ExtCaptureState {
    fn event(
        _: &mut Self,
        _: &ext_foreign_toplevel_list_v1::ExtForeignToplevelListV1,
        _: ext_foreign_toplevel_list_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }

    wayland_client::event_created_child!(Self, ext_foreign_toplevel_list_v1::ExtForeignToplevelListV1, [
        ext_foreign_toplevel_list_v1::EVT_TOPLEVEL_OPCODE => (ext_foreign_toplevel_handle_v1::ExtForeignToplevelHandleV1, ()),
    ]);
}

impl Dispatch<ext_foreign_toplevel_handle_v1::ExtForeignToplevelHandleV1, ()> for ExtCaptureState {
    fn event(
        state: &mut Self,
        proxy: &ext_foreign_toplevel_handle_v1::ExtForeignToplevelHandleV1,
        event: ext_foreign_toplevel_handle_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            ext_foreign_toplevel_handle_v1::Event::AppId { app_id } => {
                if let Some(t) = state.toplevels.iter_mut().find(|t| t.handle == *proxy) {
                    t.app_id = app_id;
                } else {
                    state.toplevels.push(DiscoveredToplevel {
                        handle: proxy.clone(),
                        app_id,
                        title: String::new(),
                        identifier: String::new(),
                    });
                }
            }
            ext_foreign_toplevel_handle_v1::Event::Title { title } => {
                if let Some(t) = state.toplevels.iter_mut().find(|t| t.handle == *proxy) {
                    t.title = title;
                }
            }
            ext_foreign_toplevel_handle_v1::Event::Identifier { identifier } => {
                if let Some(t) = state.toplevels.iter_mut().find(|t| t.handle == *proxy) {
                    t.identifier = identifier;
                }
            }
            ext_foreign_toplevel_handle_v1::Event::Closed => {
                state.toplevels.retain(|t| t.handle != *proxy);
            }
            _ => {}
        }
    }
}

// --- Toplevel enumeration ---

/// Enumerate all currently visible toplevels.
///
/// Returns `Err(NoBackend)` if `ext_foreign_toplevel_list_v1` is not available
/// (e.g. on GNOME/Mutter).
pub fn enumerate_toplevels() -> Result<Vec<ToplevelInfo>, CaptureError> {
    let conn = Connection::connect_to_env()?;
    let mut event_queue = conn.new_event_queue::<ExtCaptureState>();
    let qh = event_queue.handle();

    let mut state = ExtCaptureState {
        shm: None,
        output_source_manager: None,
        toplevel_source_manager: None,
        copy_capture_manager: None,
        toplevel_list: None,
        outputs: Vec::new(),
        toplevels: Vec::new(),
        source: None,
        session: None,
        shm_pool: None,
        buffer_width: 0,
        buffer_height: 0,
        shm_format: None,
        session_ready: false,
        frame_ready: false,
        frame_failed: false,
        damage_rects: Vec::new(),
        timestamp_ns: 0,
        stopped: false,
    };

    conn.display().get_registry(&qh, ());
    event_queue.roundtrip(&mut state)?;

    // Only the toplevel list protocol is needed for enumeration.
    // Capture protocols (source manager, copy capture) are checked later
    // when actually creating a capture backend.
    if state.toplevel_list.is_none() {
        return Err(CaptureError::NoBackend);
    }

    // Second round-trip to receive toplevel info (app_id, title, identifier).
    event_queue.roundtrip(&mut state)?;

    Ok(state
        .toplevels
        .iter()
        .map(|t| ToplevelInfo {
            app_id: t.app_id.clone(),
            title: t.title.clone(),
            identifier: t.identifier.clone(),
        })
        .collect())
}

// --- Probe ---

struct ExtCaptureProbe {
    found: bool,
}

impl Dispatch<wl_registry::WlRegistry, ()> for ExtCaptureProbe {
    fn event(
        state: &mut Self,
        _registry: &wl_registry::WlRegistry,
        event: wl_registry::Event,
        _: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        if let wl_registry::Event::Global { interface, .. } = event
            && interface == "ext_image_copy_capture_manager_v1"
        {
            state.found = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- CaptureSource construction and matching ---

    #[test]
    fn capture_source_output_none() {
        let src = CaptureSource::Output(None);
        assert!(matches!(src, CaptureSource::Output(None)));
    }

    #[test]
    fn capture_source_output_named() {
        let src = CaptureSource::Output(Some("HDMI-A-1".to_string()));
        if let CaptureSource::Output(Some(name)) = &src {
            assert_eq!(name, "HDMI-A-1");
        } else {
            panic!("expected CaptureSource::Output(Some)");
        }
    }

    #[test]
    fn capture_source_toplevel() {
        let src = CaptureSource::Toplevel("org.mozilla.firefox".to_string());
        if let CaptureSource::Toplevel(app_id) = &src {
            assert_eq!(app_id, "org.mozilla.firefox");
        } else {
            panic!("expected CaptureSource::Toplevel");
        }
    }

    #[test]
    fn capture_source_new_toplevel() {
        let known = vec!["id1".to_string(), "id2".to_string()];
        let src = CaptureSource::NewToplevel {
            known_identifiers: known.clone(),
        };
        if let CaptureSource::NewToplevel { known_identifiers } = &src {
            assert_eq!(known_identifiers.len(), 2);
            assert_eq!(known_identifiers[0], "id1");
        } else {
            panic!("expected CaptureSource::NewToplevel");
        }
    }

    #[test]
    fn capture_source_new_toplevel_empty_known() {
        let src = CaptureSource::NewToplevel {
            known_identifiers: Vec::new(),
        };
        if let CaptureSource::NewToplevel { known_identifiers } = &src {
            assert!(known_identifiers.is_empty());
        } else {
            panic!("expected CaptureSource::NewToplevel");
        }
    }

    // --- ToplevelInfo ---

    #[test]
    fn toplevel_info_construction() {
        let info = ToplevelInfo {
            app_id: "org.gnome.Terminal".to_string(),
            title: "Terminal".to_string(),
            identifier: "abc-123".to_string(),
        };
        assert_eq!(info.app_id, "org.gnome.Terminal");
        assert_eq!(info.title, "Terminal");
        assert_eq!(info.identifier, "abc-123");
    }

    #[test]
    fn toplevel_info_debug() {
        let info = ToplevelInfo {
            app_id: "app".to_string(),
            title: "Title".to_string(),
            identifier: "id".to_string(),
        };
        let dbg = format!("{:?}", info);
        assert!(dbg.contains("app"));
        assert!(dbg.contains("Title"));
        assert!(dbg.contains("id"));
    }

    #[test]
    fn toplevel_info_clone() {
        let info = ToplevelInfo {
            app_id: "x".to_string(),
            title: "y".to_string(),
            identifier: "z".to_string(),
        };
        let cloned = info.clone();
        assert_eq!(info, cloned);
    }

    #[test]
    fn toplevel_info_eq() {
        let a = ToplevelInfo {
            app_id: "a".to_string(),
            title: "t".to_string(),
            identifier: "i".to_string(),
        };
        let b = ToplevelInfo {
            app_id: "a".to_string(),
            title: "t".to_string(),
            identifier: "i".to_string(),
        };
        let c = ToplevelInfo {
            app_id: "b".to_string(),
            title: "t".to_string(),
            identifier: "i".to_string(),
        };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn toplevel_info_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        let info = ToplevelInfo {
            app_id: "a".to_string(),
            title: "t".to_string(),
            identifier: "i".to_string(),
        };
        set.insert(info.clone());
        assert!(set.contains(&info));
        // Different identifier should be different entry
        let info2 = ToplevelInfo {
            app_id: "a".to_string(),
            title: "t".to_string(),
            identifier: "j".to_string(),
        };
        set.insert(info2.clone());
        assert_eq!(set.len(), 2);
    }

    // --- Error paths without Wayland ---

    #[test]
    fn ext_backend_new_without_wayland_returns_error() {
        if std::env::var("WAYLAND_DISPLAY").is_ok() {
            return;
        }
        let result = ExtImageCaptureBackend::new(CaptureSource::Output(None));
        assert!(result.is_err());
    }

    #[test]
    fn ext_backend_new_with_named_output_without_wayland() {
        if std::env::var("WAYLAND_DISPLAY").is_ok() {
            return;
        }
        let result =
            ExtImageCaptureBackend::new(CaptureSource::Output(Some("HDMI-A-1".to_string())));
        assert!(result.is_err());
    }

    #[test]
    fn ext_backend_new_toplevel_without_wayland() {
        if std::env::var("WAYLAND_DISPLAY").is_ok() {
            return;
        }
        let result =
            ExtImageCaptureBackend::new(CaptureSource::Toplevel("org.example.app".to_string()));
        assert!(result.is_err());
    }

    #[test]
    fn ext_backend_new_new_toplevel_without_wayland() {
        if std::env::var("WAYLAND_DISPLAY").is_ok() {
            return;
        }
        let result = ExtImageCaptureBackend::new(CaptureSource::NewToplevel {
            known_identifiers: vec!["old".to_string()],
        });
        assert!(result.is_err());
    }

    #[test]
    fn enumerate_toplevels_without_wayland_returns_error() {
        if std::env::var("WAYLAND_DISPLAY").is_ok() {
            return;
        }
        let result = enumerate_toplevels();
        assert!(result.is_err());
    }

    #[test]
    fn is_available_without_wayland_returns_false() {
        if std::env::var("WAYLAND_DISPLAY").is_ok() {
            return;
        }
        // Can't call is_available without a Connection, and Connection fails without Wayland.
        // This tests the connect_to_env path.
        let conn = Connection::connect_to_env();
        assert!(conn.is_err());
    }
}

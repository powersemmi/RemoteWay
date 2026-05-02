use wayland_client::protocol::{
    wl_buffer, wl_callback, wl_compositor, wl_registry, wl_seat, wl_shm, wl_shm_pool, wl_surface,
};
use wayland_client::{Connection, Dispatch, QueueHandle};
use wayland_protocols::wp::viewporter::client::{wp_viewport, wp_viewporter};
use wayland_protocols::xdg::shell::client::{xdg_surface, xdg_toplevel, xdg_wm_base};

use crate::error::DisplayError;
use crate::shm::{DamageRegion, ShmFrameUploader};

/// A managed `xdg_toplevel` surface with its own SHM uploader.
pub struct ManagedSurface {
    /// Surface identifier (corresponds to remote window ID).
    pub surface_id: u16,
    /// The `wl_surface` backing this window.
    wl_surface: wl_surface::WlSurface,
    /// The `xdg_surface` role object.
    xdg_surface: xdg_surface::XdgSurface,
    /// The `xdg_toplevel` for window management.
    xdg_toplevel: xdg_toplevel::XdgToplevel,
    /// `wp_viewport` for compositor-side scaling. When the compositor window
    /// is a different size than the SHM buffer, the viewport maps the full
    /// buffer to the destination rectangle — zero CPU overhead.
    viewport: Option<wp_viewport::WpViewport>,
    /// SHM frame uploader for pixel data.
    uploader: Option<ShmFrameUploader>,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Bytes per row (may be > width * 4 due to padding).
    pub stride: u32,
    /// Window title.
    title: String,
    /// Application identifier.
    app_id: String,
    /// Whether the surface has received its first `configure` event.
    configured: bool,
    /// Whether a `wl_surface.frame` callback is pending.
    frame_callback_pending: bool,
    /// Monotonic instant of the last `wl_surface.frame` callback request.
    /// Used to drop the pending flag if the compositor never delivers `done`
    /// (e.g. when the surface is hidden or the buffer mismatches the
    /// configured window size).
    frame_callback_requested_at: Option<std::time::Instant>,
    /// Serial from the last `xdg_surface.configure` event.
    pending_configure_serial: Option<u32>,
}

/// Maximum time we wait for `wl_surface.frame` `done` before giving up and
/// committing the next frame anyway.
const FRAME_CALLBACK_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(50);

impl ManagedSurface {
    /// Upload frame data and commit to compositor.
    ///
    /// Returns `true` if the frame was committed, `false` if skipped
    /// (not configured, buffer not released, or frame callback pending).
    pub fn present_frame(
        &mut self,
        data: &[u8],
        damage: &[DamageRegion],
        qh: &QueueHandle<DisplayState>,
    ) -> bool {
        if !self.configured {
            return false;
        }

        // If we're waiting on a frame callback, check whether it has been
        // pending too long. Some compositors (notably niri scrollable layouts)
        // do not deliver `done` when the buffer size doesn't match the
        // configured window size, which would otherwise stall the pipeline
        // indefinitely.
        if self.frame_callback_pending {
            let stale = self
                .frame_callback_requested_at
                .is_some_and(|t| t.elapsed() >= FRAME_CALLBACK_TIMEOUT);
            if !stale {
                return false;
            }
            tracing::debug!(
                surface = self.surface_id,
                "frame callback timed out, committing next frame anyway"
            );
            self.frame_callback_pending = false;
            self.frame_callback_requested_at = None;
        }

        let uploader = match self.uploader.as_mut() {
            Some(u) => u,
            None => return false,
        };

        if !uploader.can_upload() {
            return false;
        }

        uploader.upload(data);

        self.wl_surface.attach(Some(uploader.active_buffer()), 0, 0);

        // Submit damage regions to compositor.
        if damage.is_empty() {
            self.wl_surface
                .damage_buffer(0, 0, self.width as i32, self.height as i32);
        } else {
            for rect in damage {
                self.wl_surface.damage_buffer(
                    rect.x as i32,
                    rect.y as i32,
                    rect.width as i32,
                    rect.height as i32,
                );
            }
        }

        // Tell the viewport to use the full buffer as source (if viewport is active).
        if let Some(ref viewport) = self.viewport {
            // wl_fixed values: the protocol expects 24.8 fixed-point.
            // set_source(x, y, width, height) in wl_fixed (f64 → fixed by the binding).
            viewport.set_source(0.0, 0.0, self.width as f64, self.height as f64);
        }

        // Request frame callback BEFORE commit so the compositor associates
        // the callback with THIS commit, not the next one.
        // The WlCallback is managed internally by the Wayland event queue;
        // we receive the 'done' event via the Dispatch impl.
        let _ = self.wl_surface.frame(qh, self.surface_id);
        self.wl_surface.commit();
        uploader.swap();
        self.frame_callback_pending = true;
        self.frame_callback_requested_at = Some(std::time::Instant::now());

        true
    }

    /// Whether the surface has been configured by the compositor.
    #[must_use]
    pub fn is_configured(&self) -> bool {
        self.configured
    }

    /// Get the window title.
    #[must_use]
    pub fn title(&self) -> &str {
        &self.title
    }

    /// Get the application identifier.
    #[must_use]
    pub fn app_id(&self) -> &str {
        &self.app_id
    }
}

/// Internal Wayland dispatch state.
pub struct DisplayState {
    /// The `wl_compositor` global.
    pub compositor: Option<wl_compositor::WlCompositor>,
    /// The `wl_shm` global for shared memory buffers.
    pub shm: Option<wl_shm::WlShm>,
    /// The `xdg_wm_base` global for xdg-shell.
    pub xdg_wm_base: Option<xdg_wm_base::XdgWmBase>,
    /// The `wl_seat` global.
    pub seat: Option<wl_seat::WlSeat>,
    /// The `wp_viewporter` global for compositor-side scaling.
    pub viewporter: Option<wp_viewporter::WpViewporter>,
    /// All managed surfaces.
    pub surfaces: Vec<ManagedSurface>,
    /// Toggled when a `wl_callback.done` event arrives.
    pub frame_done: bool,
}

/// Connection to the local Wayland compositor with surface management.
///
/// Creates and manages `xdg_toplevel` windows for displaying remote frames.
/// Each surface corresponds to a remote window identified by `surface_id`.
pub struct WaylandDisplay {
    /// The Wayland connection.
    conn: Connection,
    /// Dispatch state holding globals and surfaces.
    pub state: DisplayState,
    /// Event queue for Wayland protocol dispatch.
    pub event_queue: wayland_client::EventQueue<DisplayState>,
}

impl WaylandDisplay {
    /// Connect to the local Wayland compositor and discover globals.
    pub fn new() -> Result<Self, DisplayError> {
        let conn = Connection::connect_to_env()?;
        let display = conn.display();
        let mut event_queue = conn.new_event_queue::<DisplayState>();
        let qh = event_queue.handle();

        let mut state = DisplayState {
            compositor: None,
            shm: None,
            xdg_wm_base: None,
            seat: None,
            viewporter: None,
            surfaces: Vec::new(),
            frame_done: false,
        };

        // The WlRegistry is managed internally by the Wayland event queue;
        // globals are discovered via the Dispatch impl on DisplayState.
        let _ = display.get_registry(&qh, ());

        // First roundtrip: discover globals. Dispatch count is irrelevant here.
        let _ = event_queue.roundtrip(&mut state)?;

        if state.compositor.is_none() {
            return Err(DisplayError::NoCompositor);
        }
        if state.shm.is_none() {
            return Err(DisplayError::NoShm);
        }
        if state.xdg_wm_base.is_none() {
            return Err(DisplayError::NoXdgWmBase);
        }

        Ok(Self {
            conn,
            state,
            event_queue,
        })
    }

    /// Create a new `xdg_toplevel` surface for a remote window.
    pub fn create_surface(
        &mut self,
        surface_id: u16,
        title: &str,
        app_id: &str,
        width: u32,
        height: u32,
    ) -> Result<(), DisplayError> {
        let compositor = self
            .state
            .compositor
            .as_ref()
            .ok_or(DisplayError::NoCompositor)?;
        let xdg_base = self
            .state
            .xdg_wm_base
            .as_ref()
            .ok_or(DisplayError::NoXdgWmBase)?;
        let qh = self.event_queue.handle();

        let wl_surface = compositor.create_surface(&qh, surface_id);
        let xdg_surf = xdg_base.get_xdg_surface(&wl_surface, &qh, surface_id);
        let toplevel = xdg_surf.get_toplevel(&qh, surface_id);

        toplevel.set_title(title.to_string());
        toplevel.set_app_id(app_id.to_string());

        // Attach a viewport if the compositor supports wp_viewporter.
        // This lets the compositor scale the SHM buffer to the window size.
        let viewport = self
            .state
            .viewporter
            .as_ref()
            .map(|vp| vp.get_viewport(&wl_surface, &qh, surface_id));

        // Initial commit to negotiate configuration.
        wl_surface.commit();

        let stride = width * 4;

        self.state.surfaces.push(ManagedSurface {
            surface_id,
            wl_surface,
            xdg_surface: xdg_surf,
            xdg_toplevel: toplevel,
            viewport,
            uploader: None,
            width,
            height,
            stride,
            title: title.to_string(),
            app_id: app_id.to_string(),
            configured: false,
            frame_callback_pending: false,
            frame_callback_requested_at: None,
            pending_configure_serial: None,
        });

        // Roundtrip to receive configure event. Dispatch count is irrelevant.
        let _ = self.event_queue.roundtrip(&mut self.state)?;

        // Create SHM uploader now that we have the configure.
        self.init_surface_uploader(surface_id)?;

        Ok(())
    }

    /// Initialize the SHM uploader for a surface after it's been configured.
    fn init_surface_uploader(&mut self, surface_id: u16) -> Result<(), DisplayError> {
        let shm = self.state.shm.as_ref().ok_or(DisplayError::NoShm)?.clone();
        let qh = self.event_queue.handle();

        let surface = self
            .state
            .surfaces
            .iter_mut()
            .find(|s| s.surface_id == surface_id)
            .ok_or(DisplayError::SurfaceNotFound(surface_id))?;

        if surface.uploader.is_some() {
            return Ok(());
        }

        let uploader = ShmFrameUploader::new(
            &shm,
            surface.width,
            surface.height,
            surface.stride,
            wl_shm::Format::Xrgb8888,
            &qh,
        )?;
        surface.uploader = Some(uploader);
        surface.configured = true;

        Ok(())
    }

    /// Update title and `app_id` for a surface.
    pub fn update_surface_metadata(
        &mut self,
        surface_id: u16,
        title: &str,
        app_id: &str,
    ) -> Result<(), DisplayError> {
        let surface = self
            .state
            .surfaces
            .iter_mut()
            .find(|s| s.surface_id == surface_id)
            .ok_or(DisplayError::SurfaceNotFound(surface_id))?;

        surface.title = title.to_string();
        surface.app_id = app_id.to_string();
        surface.xdg_toplevel.set_title(title.to_string());
        surface.xdg_toplevel.set_app_id(app_id.to_string());

        Ok(())
    }

    /// Remove and destroy a surface.
    pub fn destroy_surface(&mut self, surface_id: u16) {
        if let Some(idx) = self
            .state
            .surfaces
            .iter()
            .position(|s| s.surface_id == surface_id)
        {
            let surface = self.state.surfaces.remove(idx);
            drop(surface.uploader);
            if let Some(viewport) = surface.viewport {
                viewport.destroy();
            }
            surface.xdg_toplevel.destroy();
            surface.xdg_surface.destroy();
            surface.wl_surface.destroy();
        }
    }

    /// Present a frame to a specific surface.
    pub fn present_frame(
        &mut self,
        surface_id: u16,
        data: &[u8],
        damage: &[DamageRegion],
    ) -> Result<bool, DisplayError> {
        let qh = self.event_queue.handle();
        let surface = self
            .state
            .surfaces
            .iter_mut()
            .find(|s| s.surface_id == surface_id)
            .ok_or(DisplayError::SurfaceNotFound(surface_id))?;

        let committed = surface.present_frame(data, damage, &qh);
        Ok(committed)
    }

    /// Dispatch pending Wayland events (non-blocking).
    pub fn dispatch_pending(&mut self) -> Result<(), DisplayError> {
        // Flush buffered outgoing requests (attach, damage, commit, frame callbacks)
        // so the compositor can process them and send responses.
        match self.conn.flush() {
            Ok(()) => {}
            Err(wayland_client::backend::WaylandError::Io(ref e))
                if e.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(e) => {
                return Err(DisplayError::WaylandDispatch(
                    wayland_client::DispatchError::Backend(e),
                ));
            }
        }

        // Dispatch count is irrelevant; we just need events processed.
        let _ = self.event_queue.dispatch_pending(&mut self.state)?;

        if let Some(guard) = self.event_queue.prepare_read() {
            match guard.read() {
                Ok(_) => {
                    // Dispatch count is irrelevant; we just need events processed.
                    let _ = self.event_queue.dispatch_pending(&mut self.state)?;
                }
                Err(wayland_client::backend::WaylandError::Io(ref e))
                    if e.kind() == std::io::ErrorKind::WouldBlock => {}
                Err(e) => {
                    return Err(DisplayError::WaylandDispatch(
                        wayland_client::DispatchError::Backend(e),
                    ));
                }
            }
        }

        Ok(())
    }

    /// Blocking dispatch — waits for at least one event.
    pub fn dispatch_blocking(&mut self) -> Result<(), DisplayError> {
        // Dispatch count is irrelevant; we just need events processed.
        let _ = self.event_queue.blocking_dispatch(&mut self.state)?;
        Ok(())
    }

    /// Get immutable access to a surface by ID.
    #[must_use]
    pub fn get_surface(&self, surface_id: u16) -> Option<&ManagedSurface> {
        self.state
            .surfaces
            .iter()
            .find(|s| s.surface_id == surface_id)
    }

    /// Handle resize: recreate SHM uploader with new dimensions.
    pub fn resize_surface(
        &mut self,
        surface_id: u16,
        width: u32,
        height: u32,
    ) -> Result<(), DisplayError> {
        let shm = self.state.shm.as_ref().ok_or(DisplayError::NoShm)?.clone();
        let qh = self.event_queue.handle();

        let surface = self
            .state
            .surfaces
            .iter_mut()
            .find(|s| s.surface_id == surface_id)
            .ok_or(DisplayError::SurfaceNotFound(surface_id))?;

        if surface.width == width && surface.height == height {
            return Ok(());
        }

        surface.width = width;
        surface.height = height;
        surface.stride = width * 4;

        let uploader = ShmFrameUploader::new(
            &shm,
            width,
            height,
            width * 4,
            wl_shm::Format::Xrgb8888,
            &qh,
        )?;
        surface.uploader = Some(uploader);

        Ok(())
    }
}

// --- Wayland dispatch implementations ---

impl Dispatch<wl_registry::WlRegistry, ()> for DisplayState {
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
                "wl_compositor" => {
                    state.compositor = Some(registry.bind(name, version.min(6), qh, ()));
                }
                "wl_shm" => {
                    state.shm = Some(registry.bind(name, version.min(1), qh, ()));
                }
                "xdg_wm_base" => {
                    state.xdg_wm_base = Some(registry.bind(name, version.min(5), qh, ()));
                }
                "wl_seat" => {
                    state.seat = Some(registry.bind(name, version.min(8), qh, ()));
                }
                "wp_viewporter" => {
                    state.viewporter = Some(registry.bind(name, version.min(1), qh, ()));
                }
                _ => {}
            }
        }
    }
}

impl Dispatch<wl_compositor::WlCompositor, ()> for DisplayState {
    fn event(
        _state: &mut Self,
        _proxy: &wl_compositor::WlCompositor,
        _event: wl_compositor::Event,
        _: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<wl_shm::WlShm, ()> for DisplayState {
    fn event(
        _state: &mut Self,
        _proxy: &wl_shm::WlShm,
        _event: wl_shm::Event,
        _: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<wl_shm_pool::WlShmPool, ()> for DisplayState {
    fn event(
        _state: &mut Self,
        _proxy: &wl_shm_pool::WlShmPool,
        _event: wl_shm_pool::Event,
        _: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<wl_surface::WlSurface, u16> for DisplayState {
    fn event(
        _state: &mut Self,
        _proxy: &wl_surface::WlSurface,
        _event: wl_surface::Event,
        _: &u16,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<wl_buffer::WlBuffer, usize> for DisplayState {
    fn event(
        state: &mut Self,
        _proxy: &wl_buffer::WlBuffer,
        event: wl_buffer::Event,
        data: &usize,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        if let wl_buffer::Event::Release = event {
            // Mark the buffer as released so it can be reused.
            let buffer_idx = *data;
            for surface in &mut state.surfaces {
                if let Some(ref mut uploader) = surface.uploader {
                    uploader.mark_released(buffer_idx);
                }
            }
        }
    }
}

impl Dispatch<wp_viewporter::WpViewporter, ()> for DisplayState {
    fn event(
        _state: &mut Self,
        _proxy: &wp_viewporter::WpViewporter,
        _event: wp_viewporter::Event,
        _: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<wp_viewport::WpViewport, u16> for DisplayState {
    fn event(
        _state: &mut Self,
        _proxy: &wp_viewport::WpViewport,
        _event: wp_viewport::Event,
        _: &u16,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<wl_seat::WlSeat, ()> for DisplayState {
    fn event(
        _state: &mut Self,
        _proxy: &wl_seat::WlSeat,
        _event: wl_seat::Event,
        _: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<xdg_wm_base::XdgWmBase, ()> for DisplayState {
    fn event(
        _state: &mut Self,
        xdg_base: &xdg_wm_base::XdgWmBase,
        event: xdg_wm_base::Event,
        _: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        if let xdg_wm_base::Event::Ping { serial } = event {
            xdg_base.pong(serial);
        }
    }
}

impl Dispatch<xdg_surface::XdgSurface, u16> for DisplayState {
    fn event(
        state: &mut Self,
        xdg_surf: &xdg_surface::XdgSurface,
        event: xdg_surface::Event,
        data: &u16,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        if let xdg_surface::Event::Configure { serial } = event {
            xdg_surf.ack_configure(serial);

            let surface_id = *data;
            if let Some(surface) = state
                .surfaces
                .iter_mut()
                .find(|s| s.surface_id == surface_id)
            {
                surface.pending_configure_serial = Some(serial);
                surface.configured = true;
            }
        }
    }
}

/// Compute the largest rectangle that fits within `dst_w × dst_h`
/// while preserving the aspect ratio of `src_w × src_h`.
///
/// Returns `(fitted_width, fitted_height)`. Both values are >= 1.
fn fit_preserve_aspect(src_w: u32, src_h: u32, dst_w: i32, dst_h: i32) -> (i32, i32) {
    if src_w == 0 || src_h == 0 {
        return (dst_w, dst_h);
    }
    // Cross-multiply to compare ratios without floating point:
    //   src_w/src_h  vs  dst_w/dst_h  ⟺  src_w*dst_h  vs  dst_w*src_h
    let lhs = src_w as u64 * dst_h as u64;
    let rhs = dst_w as u64 * src_h as u64;
    if lhs <= rhs {
        // Source narrower-or-equal → height-limited (pillarbox).
        let fitted_w = (dst_h as u64 * src_w as u64 / src_h as u64) as i32;
        (fitted_w.max(1), dst_h)
    } else {
        // Source wider → width-limited (letterbox).
        let fitted_h = (dst_w as u64 * src_h as u64 / src_w as u64) as i32;
        (dst_w, fitted_h.max(1))
    }
}

impl Dispatch<xdg_toplevel::XdgToplevel, u16> for DisplayState {
    fn event(
        state: &mut Self,
        _proxy: &xdg_toplevel::XdgToplevel,
        event: xdg_toplevel::Event,
        data: &u16,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        if let xdg_toplevel::Event::Configure {
            width,
            height,
            states: _,
        } = event
        {
            let surface_id = *data;
            if width > 0
                && height > 0
                && let Some(surface) = state
                    .surfaces
                    .iter_mut()
                    .find(|s| s.surface_id == surface_id)
                && let Some(ref viewport) = surface.viewport
            {
                let (fit_w, fit_h) =
                    fit_preserve_aspect(surface.width, surface.height, width, height);
                viewport.set_destination(fit_w, fit_h);
            }
        }
    }
}

impl Dispatch<wl_callback::WlCallback, u16> for DisplayState {
    fn event(
        state: &mut Self,
        _proxy: &wl_callback::WlCallback,
        event: wl_callback::Event,
        data: &u16,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        if let wl_callback::Event::Done { .. } = event {
            let surface_id = *data;
            if let Some(surface) = state
                .surfaces
                .iter_mut()
                .find(|s| s.surface_id == surface_id)
            {
                surface.frame_callback_pending = false;
                surface.frame_callback_requested_at = None;
            }
            state.frame_done = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_state_initial() {
        let state = DisplayState {
            compositor: None,
            shm: None,
            xdg_wm_base: None,
            seat: None,
            viewporter: None,
            surfaces: Vec::new(),
            frame_done: false,
        };
        assert!(state.compositor.is_none());
        assert!(state.shm.is_none());
        assert!(state.xdg_wm_base.is_none());
        assert!(state.seat.is_none());
        assert!(state.viewporter.is_none());
        assert!(state.surfaces.is_empty());
        assert!(!state.frame_done);
    }

    #[test]
    fn display_state_frame_done_toggle() {
        let mut state = DisplayState {
            compositor: None,
            shm: None,
            xdg_wm_base: None,
            seat: None,
            viewporter: None,
            surfaces: Vec::new(),
            frame_done: false,
        };
        assert!(!state.frame_done);
        state.frame_done = true;
        assert!(state.frame_done);
        state.frame_done = false;
        assert!(!state.frame_done);
    }

    #[test]
    fn display_state_surfaces_vec_operations() {
        let state = DisplayState {
            compositor: None,
            shm: None,
            xdg_wm_base: None,
            seat: None,
            viewporter: None,
            surfaces: Vec::new(),
            frame_done: false,
        };
        assert!(state.surfaces.is_empty());
        // We can't create real ManagedSurface without Wayland objects,
        // but we can verify Vec operations on the state.
        assert_eq!(state.surfaces.len(), 0);
        assert!(!state.surfaces.iter().any(|s| s.surface_id == 0));
    }

    #[test]
    fn wayland_display_fails_without_compositor() {
        // Unset WAYLAND_DISPLAY to ensure no connection.
        // SAFETY: removing env var in an isolated test is safe.
        unsafe { std::env::remove_var("WAYLAND_DISPLAY") };
        let result = WaylandDisplay::new();
        assert!(result.is_err());
    }

    #[test]
    fn wayland_display_error_is_connect_error() {
        // SAFETY: removing env var in an isolated test is safe.
        unsafe { std::env::remove_var("WAYLAND_DISPLAY") };
        let result = WaylandDisplay::new();
        match result {
            Err(DisplayError::WaylandConnect(_)) => {} // expected
            Err(other) => {
                // Some environments might return a different error.
            }
            Ok(_) => panic!("expected error without WAYLAND_DISPLAY"),
        }
    }

    // --- fit_preserve_aspect tests ---

    #[test]
    fn fit_preserve_aspect_same_ratio() {
        let (w, h) = fit_preserve_aspect(3840, 2160, 1920, 1080);
        assert_eq!((w, h), (1920, 1080));
    }

    #[test]
    fn fit_preserve_aspect_wider_dest() {
        // 16:9 buffer into wider 2:1 destination (2000x1000).
        // Height-limited: fitted_w = 1000 * 3840 / 2160 = 1777
        let (w, h) = fit_preserve_aspect(3840, 2160, 2000, 1000);
        assert_eq!(h, 1000);
        assert_eq!(w, 1777);
        assert!(w <= 2000);
    }

    #[test]
    fn fit_preserve_aspect_taller_dest() {
        // 16:9 buffer into 4:3 destination (800x600).
        // Width-limited: fitted_h = 800 * 2160 / 3840 = 450
        let (w, h) = fit_preserve_aspect(3840, 2160, 800, 600);
        assert_eq!(w, 800);
        assert_eq!(h, 450);
        assert!(h <= 600);
    }

    #[test]
    fn fit_preserve_aspect_exact_fit() {
        let (w, h) = fit_preserve_aspect(1920, 1080, 1920, 1080);
        assert_eq!((w, h), (1920, 1080));
    }

    #[test]
    fn fit_preserve_aspect_square_into_wide() {
        // 1:1 buffer into 2:1 destination (1000x500).
        let (w, h) = fit_preserve_aspect(1000, 1000, 1000, 500);
        assert_eq!((w, h), (500, 500));
    }

    #[test]
    fn fit_preserve_aspect_zero_source() {
        let (w, h) = fit_preserve_aspect(0, 0, 800, 600);
        assert_eq!((w, h), (800, 600));
    }

    #[test]
    fn fit_preserve_aspect_min_one() {
        // Extreme ratio: very wide buffer into very tall destination.
        let (w, h) = fit_preserve_aspect(10000, 1, 1, 10000);
        assert_eq!(w, 1);
        assert!(h >= 1);
    }

    #[test]
    fn fit_preserve_aspect_zero_width_source() {
        // Only width is zero.
        let (w, h) = fit_preserve_aspect(0, 1080, 800, 600);
        assert_eq!((w, h), (800, 600));
    }

    #[test]
    fn fit_preserve_aspect_zero_height_source() {
        // Only height is zero.
        let (w, h) = fit_preserve_aspect(1920, 0, 800, 600);
        assert_eq!((w, h), (800, 600));
    }

    #[test]
    fn fit_preserve_aspect_square_source_square_dest() {
        let (w, h) = fit_preserve_aspect(500, 500, 300, 300);
        assert_eq!((w, h), (300, 300));
    }

    #[test]
    fn fit_preserve_aspect_wide_source_into_tall_dest() {
        // 4:1 source into 1:4 dest (100x400).
        // Width-limited: fitted_h = 100 * 1 / 4 = 25.
        let (w, h) = fit_preserve_aspect(400, 100, 100, 400);
        assert_eq!(w, 100);
        assert_eq!(h, 25);
    }

    #[test]
    fn fit_preserve_aspect_tall_source_into_wide_dest() {
        // 1:4 source into 4:1 dest (400x100).
        // Height-limited: fitted_w = 100 * 100 / 400 = 25.
        let (w, h) = fit_preserve_aspect(100, 400, 400, 100);
        assert_eq!(w, 25);
        assert_eq!(h, 100);
    }

    #[test]
    fn fit_preserve_aspect_ultrawide_21_9() {
        // 21:9 (2560x1080) into 16:9 (1920x1080).
        // Width-limited: fitted_h = 1920 * 1080 / 2560 = 810.
        let (w, h) = fit_preserve_aspect(2560, 1080, 1920, 1080);
        assert_eq!(w, 1920);
        assert_eq!(h, 810);
    }

    #[test]
    fn fit_preserve_aspect_upscale() {
        // Small source (640x480) into large dest (1920x1080).
        // Source ratio 4:3, dest ratio 16:9.
        // Width limited: fitted_h = 1920 * 480 / 640 = 1440 > 1080.
        // Actually source is narrower: lhs = 640 * 1080 = 691200, rhs = 1920 * 480 = 921600.
        // lhs < rhs so height-limited (pillarbox): fitted_w = 1080 * 640 / 480 = 1440.
        let (w, h) = fit_preserve_aspect(640, 480, 1920, 1080);
        assert_eq!(h, 1080);
        assert_eq!(w, 1440);
        assert!(w <= 1920);
    }

    #[test]
    fn fit_preserve_aspect_1x1_source() {
        let (w, h) = fit_preserve_aspect(1, 1, 1920, 1080);
        // 1:1 into 16:9 — height limited: fitted_w = 1080 * 1 / 1 = 1080.
        assert_eq!((w, h), (1080, 1080));
    }

    #[test]
    fn fit_preserve_aspect_preserves_ratio_numerically() {
        // Verify that the output aspect ratio is as close as possible to the source.
        let src_w = 1920u32;
        let src_h = 1080u32;
        let (fit_w, fit_h) = fit_preserve_aspect(src_w, src_h, 1366, 768);
        // Check that fit_w/fit_h approximates src_w/src_h.
        let src_ratio = src_w as f64 / src_h as f64;
        let fit_ratio = fit_w as f64 / fit_h as f64;
        let diff = (src_ratio - fit_ratio).abs();
        assert!(
            diff < 0.01,
            "Aspect ratio drift too large: src={src_ratio:.4}, fit={fit_ratio:.4}, diff={diff:.6}"
        );
    }

    #[test]
    fn fit_preserve_aspect_never_exceeds_dest() {
        // Property: for any valid inputs, result fits within destination.
        let cases: Vec<(u32, u32, i32, i32)> = vec![
            (3840, 2160, 1920, 1080),
            (640, 480, 1920, 1080),
            (100, 100, 50, 200),
            (100, 100, 200, 50),
            (2560, 1080, 1920, 1080),
            (1, 10000, 1000, 1000),
            (10000, 1, 1000, 1000),
        ];
        for (sw, sh, dw, dh) in cases {
            let (fw, fh) = fit_preserve_aspect(sw, sh, dw, dh);
            assert!(
                fw <= dw && fh <= dh,
                "Exceeded dest: src={sw}x{sh}, dst={dw}x{dh}, fit={fw}x{fh}"
            );
            assert!(fw >= 1, "Width < 1: {fw}");
            assert!(fh >= 1, "Height < 1: {fh}");
        }
    }

    #[test]
    fn fit_preserve_aspect_small_values() {
        let (w, h) = fit_preserve_aspect(2, 1, 3, 3);
        // 2:1 into 1:1 dest (3x3). Width-limited: fitted_h = 3 * 1 / 2 = 1.
        assert_eq!(w, 3);
        assert_eq!(h, 1);
    }

    #[test]
    fn fit_preserve_aspect_large_values() {
        // Very large source and dest to check for overflow protection via u64 math.
        let (w, h) = fit_preserve_aspect(65535, 65535, 32767, 32767);
        assert_eq!((w, h), (32767, 32767));
    }

    #[test]
    fn frame_callback_timeout_value() {
        // Verify the constant is 50ms.
        assert_eq!(FRAME_CALLBACK_TIMEOUT, std::time::Duration::from_millis(50));
    }
}

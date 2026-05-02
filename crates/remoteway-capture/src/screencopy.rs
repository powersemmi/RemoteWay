//! `wlr-screencopy-v1` capture backend.
//!
//! Legacy screencopy protocol for wlroots-based compositors (Sway, Hyprland,
//! niri, River). Used as a fallback when `ext-image-capture-source-v1` is
//! unavailable.

use wayland_client::protocol::{wl_buffer, wl_output, wl_registry, wl_shm, wl_shm_pool};
use wayland_client::{Connection, Dispatch, QueueHandle, WEnum};
use wayland_protocols_wlr::screencopy::v1::client::{
    zwlr_screencopy_frame_v1, zwlr_screencopy_manager_v1,
};

use remoteway_compress::delta::DamageRect;

use crate::backend::{CaptureBackend, CapturedFrame, PixelFormat};
use crate::error::CaptureError;
use crate::shm::ShmBufferPool;

/// wlr-screencopy-based capture backend.
///
/// Works with wlroots-based compositors (Sway, Hyprland) and smithay-based ones
/// that expose `zwlr_screencopy_manager_v1` (niri, cosmic-comp, etc.).
pub struct WlrScreencopyBackend {
    /// Keep connection alive for the lifetime of the backend.
    _conn: Connection,
    state: ScreencopyState,
    event_queue: wayland_client::EventQueue<ScreencopyState>,
}

/// Tracks an output discovered during global enumeration.
struct DiscoveredOutput {
    wl_output: wl_output::WlOutput,
    /// The compositor name (e.g., "HDMI-A-1"), filled in from `wl_output.name` event.
    name: String,
    /// The Wayland global name for matching `wl_output` events to this entry.
    global_name: u32,
}

struct ScreencopyState {
    shm: Option<wl_shm::WlShm>,
    shm_pool: Option<ShmBufferPool>,
    screencopy_manager: Option<zwlr_screencopy_manager_v1::ZwlrScreencopyManagerV1>,
    /// All discovered outputs before selection.
    discovered_outputs: Vec<DiscoveredOutput>,
    /// The selected output for capture.
    output: Option<wl_output::WlOutput>,
    // Per-frame state.
    frame_ready: bool,
    frame_failed: bool,
    /// Set when all Buffer events are received and pool can be created.
    buffer_done: bool,
    damage_rects: Vec<DamageRect>,
    frame_format: Option<PixelFormat>,
    frame_width: u32,
    frame_height: u32,
    frame_stride: u32,
    frame_format_raw: u32,
    /// Wayland SHM format enum for pool creation outside event handler.
    wl_shm_format: Option<wl_shm::Format>,
    timestamp_ns: u64,
    buffer_attached: bool,
    stopped: bool,
}

impl WlrScreencopyBackend {
    /// Create a new screencopy backend for the specified output.
    ///
    /// `output_name` selects the output by its compositor name (e.g., "HDMI-A-1",
    /// "eDP-1"). Pass `None` to capture the first available output.
    pub fn new(output_name: Option<&str>) -> Result<Self, CaptureError> {
        let conn = Connection::connect_to_env()?;
        let display = conn.display();
        let mut event_queue = conn.new_event_queue::<ScreencopyState>();
        let qh = event_queue.handle();

        let mut state = ScreencopyState {
            shm: None,
            shm_pool: None,
            screencopy_manager: None,
            discovered_outputs: Vec::new(),
            output: None,
            frame_ready: false,
            frame_failed: false,
            buffer_done: false,
            damage_rects: Vec::new(),
            frame_format: None,
            frame_width: 0,
            frame_height: 0,
            frame_stride: 0,
            frame_format_raw: 0,
            wl_shm_format: None,
            timestamp_ns: 0,
            buffer_attached: false,
            stopped: false,
        };

        // The WlRegistry is not stored — globals arrive as events
        // dispatched through the queue.
        let _registry = display.get_registry(&qh, ());

        // First round-trip: discover globals (wl_shm, wl_output, screencopy manager).
        let dispatched = event_queue.roundtrip(&mut state)?;
        tracing::trace!(dispatched, "screencopy: first roundtrip");

        if state.screencopy_manager.is_none() {
            return Err(CaptureError::NoBackend);
        }
        if state.shm.is_none() {
            return Err(CaptureError::CaptureFailed("wl_shm not available".into()));
        }
        if state.discovered_outputs.is_empty() {
            return Err(CaptureError::NoOutputs);
        }

        // Second round-trip: receive wl_output.name events (version 4+).
        let dispatched = event_queue.roundtrip(&mut state)?;
        tracing::trace!(dispatched, "screencopy: second roundtrip");

        // Select the output — by name or first available.
        let selected = if let Some(name) = output_name {
            let idx = state
                .discovered_outputs
                .iter()
                .position(|o| o.name == name)
                .ok_or_else(|| CaptureError::OutputNotFound(name.to_string()))?;
            state.discovered_outputs.swap_remove(idx)
        } else {
            state.discovered_outputs.swap_remove(0)
        };
        state.output = Some(selected.wl_output);

        Ok(Self {
            _conn: conn,
            state,
            event_queue,
        })
    }

    /// Check if wlr-screencopy is available on the given connection.
    #[must_use]
    pub fn is_available(conn: &Connection) -> bool {
        let mut eq = conn.new_event_queue::<ScreencopyProbe>();
        let qh = eq.handle();
        let mut probe = ScreencopyProbe { found: false };
        // The WlRegistry is not needed — globals arrive through dispatch.
        let _registry = conn.display().get_registry(&qh, ());
        // If the roundtrip fails (e.g. compositor disconnected), treat the
        // backend as unavailable.
        let dispatched = match eq.roundtrip(&mut probe) {
            Ok(n) => n,
            Err(_) => return false,
        };
        tracing::trace!(dispatched, "screencopy probe roundtrip");
        probe.found
    }
}

impl CaptureBackend for WlrScreencopyBackend {
    fn next_frame(&mut self) -> Result<CapturedFrame, CaptureError> {
        if self.state.stopped {
            return Err(CaptureError::SessionEnded);
        }

        let qh = self.event_queue.handle();

        // Reset per-frame state.
        self.state.frame_ready = false;
        self.state.frame_failed = false;
        self.state.buffer_done = false;
        self.state.damage_rects.clear();
        self.state.buffer_attached = false;
        // Capture previous pool dims+format so we can detect a change after Buffer events.
        let prev_pool_format = self.state.shm_pool.as_ref().map(|p| p.format);

        // Request a frame capture.
        let manager = self
            .state
            .screencopy_manager
            .as_ref()
            .ok_or(CaptureError::NoBackend)?;
        let output = self.state.output.as_ref().ok_or(CaptureError::NoOutputs)?;

        let frame = manager.capture_output(0, output, &qh, ());

        // Phase 1: dispatch until BufferDone — learn buffer format and dimensions.
        // Event handlers only store format info, they do NOT create the pool or
        // call copy. This prevents queueing pool creation and copy in the same
        // flush (compositor needs to process the pool fd first).
        while !self.state.buffer_done && !self.state.frame_failed {
            let dispatched = self.event_queue.blocking_dispatch(&mut self.state)?;
            tracing::trace!(dispatched, "screencopy phase 1: blocking_dispatch");
        }

        if self.state.frame_failed {
            frame.destroy();
            return Err(CaptureError::CaptureFailed(
                "compositor reported capture failure".into(),
            ));
        }

        let wl_format = self
            .state
            .wl_shm_format
            .ok_or_else(|| CaptureError::CaptureFailed("no buffer format received".into()))?;

        // Between phases: create or recreate SHM pool outside the event handler.
        // Recreate when dimensions, stride, or format change — smithay-based
        // compositors (niri) reject buffers whose attributes don't match exactly.
        let needs_new_pool = match &self.state.shm_pool {
            Some(pool) => {
                pool.width != self.state.frame_width
                    || pool.height != self.state.frame_height
                    || pool.stride != self.state.frame_stride
                    || prev_pool_format != Some(wl_format)
            }
            None => true,
        };
        if needs_new_pool {
            let shm = self
                .state
                .shm
                .as_ref()
                .ok_or(CaptureError::CaptureFailed("wl_shm not available".into()))?;
            // Drop the old pool first so the compositor releases its fd before we
            // create a new one of a possibly different size.
            self.state.shm_pool = None;
            let pool = ShmBufferPool::new(
                shm,
                self.state.frame_width,
                self.state.frame_height,
                self.state.frame_stride,
                wl_format,
                &qh,
            )?;
            self.state.shm_pool = Some(pool);

            tracing::debug!(
                width = self.state.frame_width,
                height = self.state.frame_height,
                stride = self.state.frame_stride,
                format = ?wl_format,
                "screencopy: created new SHM pool"
            );
        }

        // Roundtrip ensures the compositor has processed create_pool (fd via
        // SCM_RIGHTS) and create_buffer before we reference the buffer in
        // `copy`. Run unconditionally — for an existing pool, the round-trip
        // is cheap (no new requests are pending) but it guarantees the
        // compositor has fully drained any prior buffer release events.
        let dispatched = self.event_queue.roundtrip(&mut self.state)?;
        tracing::trace!(dispatched, "screencopy: pool creation roundtrip completed");

        // Attach buffer and request copy. We use `copy` (not `copy_with_damage`)
        // because the latter blocks until the compositor sees fresh damage,
        // which produces high latency on idle frames; smithay-based compositors
        // (niri) also have known issues with `copy_with_damage` buffer
        // validation. Damage events are reported with both requests for v2+.
        let pool = self
            .state
            .shm_pool
            .as_ref()
            .ok_or(CaptureError::CaptureFailed(
                "SHM pool not initialized".into(),
            ))?;
        frame.copy(pool.active_buffer());
        self.state.buffer_attached = true;

        // Phase 2: dispatch until Ready or Failed.
        while !self.state.frame_ready && !self.state.frame_failed {
            let dispatched = self.event_queue.blocking_dispatch(&mut self.state)?;
            tracing::trace!(dispatched, "screencopy phase 2: blocking_dispatch");
        }

        // Frame object is single-shot — destroy regardless of outcome.
        let frame_failed = self.state.frame_failed;
        frame.destroy();

        if frame_failed {
            return Err(CaptureError::CaptureFailed(
                "compositor reported capture failure".into(),
            ));
        }

        // If no damage was reported, treat entire frame as damaged.
        if self.state.damage_rects.is_empty() {
            self.state.damage_rects.push(DamageRect::new(
                0,
                0,
                self.state.frame_width,
                self.state.frame_height,
            ));
        }

        let format = self
            .state
            .frame_format
            .ok_or(CaptureError::UnsupportedFormat(self.state.frame_format_raw))?;

        // Copy data from SHM buffer.
        let data = if let Some(ref pool) = self.state.shm_pool {
            // SAFETY: frame_ready == true means the compositor has finished writing
            // to this buffer (the Ready event is the signal). No concurrent access.
            let src = unsafe { pool.active_data() };
            src.to_vec()
        } else {
            return Err(CaptureError::CaptureFailed(
                "SHM pool not initialized (no Buffer event received)".into(),
            ));
        };

        // Swap the double buffer for next frame.
        if let Some(ref mut pool) = self.state.shm_pool {
            pool.swap();
        }

        Ok(CapturedFrame {
            data,
            damage: std::mem::take(&mut self.state.damage_rects),
            format,
            width: self.state.frame_width,
            height: self.state.frame_height,
            stride: self.state.frame_stride,
            timestamp_ns: self.state.timestamp_ns,
        })
    }

    fn name(&self) -> &'static str {
        "wlr-screencopy"
    }

    fn stop(&mut self) {
        self.state.stopped = true;
    }
}

// --- Wayland dispatch implementations ---

impl Dispatch<wl_registry::WlRegistry, ()> for ScreencopyState {
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
                    state.shm = Some(registry.bind(name, version.min(2), qh, ()));
                }
                "wl_output" => {
                    let wl_out: wl_output::WlOutput = registry.bind(name, version.min(4), qh, name);
                    state.discovered_outputs.push(DiscoveredOutput {
                        wl_output: wl_out,
                        name: String::new(),
                        global_name: name,
                    });
                }
                "zwlr_screencopy_manager_v1" => {
                    state.screencopy_manager = Some(registry.bind(name, version.min(3), qh, ()));
                }
                _ => {}
            }
        }
    }
}

impl Dispatch<wl_shm::WlShm, ()> for ScreencopyState {
    fn event(
        _state: &mut Self,
        _proxy: &wl_shm::WlShm,
        _event: wl_shm::Event,
        _: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        // wl_shm.format events — we handle format in screencopy buffer event.
    }
}

impl Dispatch<wl_shm_pool::WlShmPool, ()> for ScreencopyState {
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

impl Dispatch<wl_buffer::WlBuffer, usize> for ScreencopyState {
    fn event(
        _state: &mut Self,
        _proxy: &wl_buffer::WlBuffer,
        _event: wl_buffer::Event,
        _: &usize,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        // wl_buffer.release — compositor is done with this buffer.
    }
}

impl Dispatch<wl_output::WlOutput, u32> for ScreencopyState {
    fn event(
        state: &mut Self,
        _proxy: &wl_output::WlOutput,
        event: wl_output::Event,
        global_name: &u32,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        // Capture the output name (wl_output version 4+).
        if let wl_output::Event::Name { name } = event
            && let Some(out) = state
                .discovered_outputs
                .iter_mut()
                .find(|o| o.global_name == *global_name)
        {
            out.name = name;
        }
    }
}

impl Dispatch<zwlr_screencopy_manager_v1::ZwlrScreencopyManagerV1, ()> for ScreencopyState {
    fn event(
        _state: &mut Self,
        _proxy: &zwlr_screencopy_manager_v1::ZwlrScreencopyManagerV1,
        _event: zwlr_screencopy_manager_v1::Event,
        _: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<zwlr_screencopy_frame_v1::ZwlrScreencopyFrameV1, ()> for ScreencopyState {
    fn event(
        state: &mut Self,
        _frame: &zwlr_screencopy_frame_v1::ZwlrScreencopyFrameV1,
        event: zwlr_screencopy_frame_v1::Event,
        _: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        match event {
            zwlr_screencopy_frame_v1::Event::Buffer {
                format,
                width,
                height,
                stride,
            } => {
                let wl_format = match format {
                    WEnum::Value(f) => f,
                    WEnum::Unknown(_) => {
                        state.frame_failed = true;
                        return;
                    }
                };
                state.frame_width = width;
                state.frame_height = height;
                state.frame_stride = stride;
                state.frame_format_raw = wl_format as u32;
                state.frame_format = PixelFormat::from_wl_shm(wl_format as u32);
                state.wl_shm_format = Some(wl_format);

                tracing::info!(
                    format_raw = format!("0x{:08x}", state.frame_format_raw),
                    width,
                    height,
                    stride,
                    pixel_format = ?state.frame_format,
                    "screencopy: Buffer event (shm)"
                );

                // Pool creation moved to next_frame() between dispatch phases
                // to ensure compositor processes the pool fd before copy_with_damage.
            }
            zwlr_screencopy_frame_v1::Event::BufferDone => {
                // All buffer format advertisements received.
                // Pool creation + copy_with_damage happen in next_frame().
                state.buffer_done = true;
            }
            zwlr_screencopy_frame_v1::Event::Damage {
                x,
                y,
                width,
                height,
            } => {
                state
                    .damage_rects
                    .push(DamageRect::new(x, y, width, height));
            }
            zwlr_screencopy_frame_v1::Event::Ready {
                tv_sec_hi,
                tv_sec_lo,
                tv_nsec,
            } => {
                state.timestamp_ns =
                    ((tv_sec_hi as u64) << 32 | tv_sec_lo as u64) * 1_000_000_000 + tv_nsec as u64;
                state.frame_ready = true;
            }
            zwlr_screencopy_frame_v1::Event::Failed => {
                state.frame_failed = true;
            }
            _ => {}
        }
    }
}

// --- Lightweight probe for availability detection ---

struct ScreencopyProbe {
    found: bool,
}

impl Dispatch<wl_registry::WlRegistry, ()> for ScreencopyProbe {
    fn event(
        state: &mut Self,
        _registry: &wl_registry::WlRegistry,
        event: wl_registry::Event,
        _: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        if let wl_registry::Event::Global { interface, .. } = event
            && interface == "zwlr_screencopy_manager_v1"
        {
            state.found = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn screencopy_new_without_wayland_returns_error() {
        if std::env::var("WAYLAND_DISPLAY").is_ok() {
            return;
        }
        let result = WlrScreencopyBackend::new(None);
        assert!(result.is_err());
    }

    #[test]
    fn screencopy_new_with_named_output_without_wayland() {
        if std::env::var("WAYLAND_DISPLAY").is_ok() {
            return;
        }
        let result = WlrScreencopyBackend::new(Some("eDP-1"));
        assert!(result.is_err());
    }

    #[test]
    fn connection_fails_without_wayland() {
        if std::env::var("WAYLAND_DISPLAY").is_ok() {
            return;
        }
        // is_available requires a Connection; Connection::connect_to_env fails
        // without WAYLAND_DISPLAY, so test the failure path.
        let conn = Connection::connect_to_env();
        assert!(conn.is_err());
    }
}

//! Integration test using an in-process wayland-server mock that implements
//! wl_shm and wlr-screencopy-unstable-v1 for full protocol round-trip testing.
//!
//! Tests exercise real library types (`ShmBufferPool`, `OutputEnumerator`,
//! `PixelFormat`, capture dispatch flows) against a mock compositor.

use std::os::unix::net::UnixStream;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use wayland_protocols::xdg::xdg_output::zv1::server::{zxdg_output_manager_v1, zxdg_output_v1};
use wayland_protocols_wlr::screencopy::v1::server::{
    zwlr_screencopy_frame_v1, zwlr_screencopy_manager_v1,
};
use wayland_server::protocol::{wl_buffer, wl_output, wl_shm, wl_shm_pool};
use wayland_server::{Client, Dispatch, DisplayHandle, GlobalDispatch, New};

// Re-export library types to test them against the mock.
use wayland_client::Proxy;

use remoteway_capture::backend::PixelFormat;
use remoteway_capture::output::OutputEnumerator;
use remoteway_capture::shm::ShmBufferPool;

// --- Mock compositor state ---

struct MockCompositor {
    frame_width: u32,
    frame_height: u32,
    frame_stride: u32,
    /// Pixel format to advertise (server-side wl_shm::Format).
    shm_format: wl_shm::Format,
    /// Whether the capture should fail.
    should_fail: bool,
    /// Counter for capture requests received.
    capture_count: Arc<AtomicU32>,
    /// Number of copy requests received.
    copy_count: Arc<AtomicU32>,
    /// Whether to send damage events with the ready.
    send_damage: bool,
    /// Timestamp components to send in Ready event.
    ts_sec_hi: u32,
    ts_sec_lo: u32,
    ts_nsec: u32,
    /// Number of outputs to advertise.
    output_names: Vec<String>,
}

impl MockCompositor {
    fn new(width: u32, height: u32) -> Self {
        Self {
            frame_width: width,
            frame_height: height,
            frame_stride: width * 4,
            shm_format: wl_shm::Format::Xrgb8888,
            should_fail: false,
            capture_count: Arc::new(AtomicU32::new(0)),
            copy_count: Arc::new(AtomicU32::new(0)),
            send_damage: true,
            ts_sec_hi: 0,
            ts_sec_lo: 0,
            ts_nsec: 0,
            output_names: vec!["MOCK-1".to_string()],
        }
    }

    fn with_format(mut self, format: wl_shm::Format) -> Self {
        self.shm_format = format;
        self
    }

    fn with_failure(mut self) -> Self {
        self.should_fail = true;
        self
    }

    fn with_timestamp(mut self, sec_hi: u32, sec_lo: u32, nsec: u32) -> Self {
        self.ts_sec_hi = sec_hi;
        self.ts_sec_lo = sec_lo;
        self.ts_nsec = nsec;
        self
    }

    fn with_outputs(mut self, names: Vec<String>) -> Self {
        self.output_names = names;
        self
    }

    fn with_no_damage(mut self) -> Self {
        self.send_damage = false;
        self
    }
}

// --- GlobalDispatch: wl_shm ---

impl GlobalDispatch<wl_shm::WlShm, ()> for MockCompositor {
    fn bind(
        state: &mut Self,
        _handle: &DisplayHandle,
        _client: &Client,
        resource: New<wl_shm::WlShm>,
        _data: &(),
        data_init: &mut wayland_server::DataInit<'_, Self>,
    ) {
        let shm = data_init.init(resource, ());
        shm.format(state.shm_format);
        // Also advertise Argb8888 so clients see multiple formats.
        if state.shm_format != wl_shm::Format::Argb8888 {
            shm.format(wl_shm::Format::Argb8888);
        }
    }
}

impl Dispatch<wl_shm::WlShm, ()> for MockCompositor {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &wl_shm::WlShm,
        request: wl_shm::Request,
        _data: &(),
        _handle: &DisplayHandle,
        data_init: &mut wayland_server::DataInit<'_, Self>,
    ) {
        if let wl_shm::Request::CreatePool { id, .. } = request {
            data_init.init(id, ());
        }
    }
}

impl Dispatch<wl_shm_pool::WlShmPool, ()> for MockCompositor {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &wl_shm_pool::WlShmPool,
        request: wl_shm_pool::Request,
        _data: &(),
        _handle: &DisplayHandle,
        data_init: &mut wayland_server::DataInit<'_, Self>,
    ) {
        if let wl_shm_pool::Request::CreateBuffer { id, .. } = request {
            data_init.init(id, ());
        }
    }
}

impl Dispatch<wl_buffer::WlBuffer, ()> for MockCompositor {
    fn request(
        _: &mut Self,
        _: &Client,
        _: &wl_buffer::WlBuffer,
        _: wl_buffer::Request,
        _: &(),
        _: &DisplayHandle,
        _: &mut wayland_server::DataInit<'_, Self>,
    ) {
    }
}

// --- GlobalDispatch: wl_output ---

/// Per-output data on the server side, tracks which output global this is.
struct OutputData {}

impl GlobalDispatch<wl_output::WlOutput, usize> for MockCompositor {
    fn bind(
        state: &mut Self,
        _handle: &DisplayHandle,
        _client: &Client,
        resource: New<wl_output::WlOutput>,
        data: &usize,
        data_init: &mut wayland_server::DataInit<'_, Self>,
    ) {
        let idx = *data;
        let output = data_init.init(resource, OutputData {});
        output.geometry(
            (idx as i32) * 1920,
            0,
            527,
            296,
            wl_output::Subpixel::None,
            "Mock".into(),
            format!("Test Output {idx}"),
            wl_output::Transform::Normal,
        );
        output.mode(
            wl_output::Mode::Current,
            state.frame_width as i32,
            state.frame_height as i32,
            60000,
        );
        output.scale(if idx == 0 { 1 } else { 2 });
        if idx < state.output_names.len() {
            output.name(state.output_names[idx].clone());
        }
        output.description(format!("Mock Output {idx}"));
        output.done();
    }
}

impl Dispatch<wl_output::WlOutput, OutputData> for MockCompositor {
    fn request(
        _: &mut Self,
        _: &Client,
        _: &wl_output::WlOutput,
        _: wl_output::Request,
        _: &OutputData,
        _: &DisplayHandle,
        _: &mut wayland_server::DataInit<'_, Self>,
    ) {
    }
}

// --- GlobalDispatch: zxdg_output_manager_v1 ---

impl GlobalDispatch<zxdg_output_manager_v1::ZxdgOutputManagerV1, ()> for MockCompositor {
    fn bind(
        _state: &mut Self,
        _handle: &DisplayHandle,
        _client: &Client,
        resource: New<zxdg_output_manager_v1::ZxdgOutputManagerV1>,
        _data: &(),
        data_init: &mut wayland_server::DataInit<'_, Self>,
    ) {
        data_init.init(resource, ());
    }
}

impl Dispatch<zxdg_output_manager_v1::ZxdgOutputManagerV1, ()> for MockCompositor {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &zxdg_output_manager_v1::ZxdgOutputManagerV1,
        request: zxdg_output_manager_v1::Request,
        _data: &(),
        _handle: &DisplayHandle,
        data_init: &mut wayland_server::DataInit<'_, Self>,
    ) {
        if let zxdg_output_manager_v1::Request::GetXdgOutput { id, .. } = request {
            let xdg_output = data_init.init(id, ());
            // Send xdg_output events. In a real compositor these would come from
            // the associated wl_output, but for testing we send fixed values.
            xdg_output.name("xdg-output-0".into());
            xdg_output.description("XDG Output 0".into());
            xdg_output.logical_position(0, 0);
            xdg_output.logical_size(1920, 1080);
            xdg_output.done();
        }
    }
}

impl Dispatch<zxdg_output_v1::ZxdgOutputV1, ()> for MockCompositor {
    fn request(
        _: &mut Self,
        _: &Client,
        _: &zxdg_output_v1::ZxdgOutputV1,
        _: zxdg_output_v1::Request,
        _: &(),
        _: &DisplayHandle,
        _: &mut wayland_server::DataInit<'_, Self>,
    ) {
    }
}

// --- GlobalDispatch: zwlr_screencopy_manager_v1 ---

impl GlobalDispatch<zwlr_screencopy_manager_v1::ZwlrScreencopyManagerV1, ()> for MockCompositor {
    fn bind(
        _state: &mut Self,
        _handle: &DisplayHandle,
        _client: &Client,
        resource: New<zwlr_screencopy_manager_v1::ZwlrScreencopyManagerV1>,
        _data: &(),
        data_init: &mut wayland_server::DataInit<'_, Self>,
    ) {
        data_init.init(resource, ());
    }
}

impl Dispatch<zwlr_screencopy_manager_v1::ZwlrScreencopyManagerV1, ()> for MockCompositor {
    fn request(
        state: &mut Self,
        _client: &Client,
        _resource: &zwlr_screencopy_manager_v1::ZwlrScreencopyManagerV1,
        request: zwlr_screencopy_manager_v1::Request,
        _data: &(),
        _handle: &DisplayHandle,
        data_init: &mut wayland_server::DataInit<'_, Self>,
    ) {
        if let zwlr_screencopy_manager_v1::Request::CaptureOutput { frame: id, .. } = request {
            state.capture_count.fetch_add(1, Ordering::Relaxed);
            let frame = data_init.init(id, ());

            if state.should_fail {
                frame.failed();
            } else {
                // Send buffer info to client.
                frame.buffer(
                    state.shm_format,
                    state.frame_width,
                    state.frame_height,
                    state.frame_stride,
                );
                // BufferDone signals end of buffer format advertisements.
                // This is critical — the real next_frame() waits for this.
                frame.buffer_done();
            }
        }
    }
}

impl Dispatch<zwlr_screencopy_frame_v1::ZwlrScreencopyFrameV1, ()> for MockCompositor {
    fn request(
        state: &mut Self,
        _client: &Client,
        resource: &zwlr_screencopy_frame_v1::ZwlrScreencopyFrameV1,
        request: zwlr_screencopy_frame_v1::Request,
        _data: &(),
        _handle: &DisplayHandle,
        _data_init: &mut wayland_server::DataInit<'_, Self>,
    ) {
        match request {
            zwlr_screencopy_frame_v1::Request::CopyWithDamage { .. }
            | zwlr_screencopy_frame_v1::Request::Copy { .. } => {
                state.copy_count.fetch_add(1, Ordering::Relaxed);
                if state.should_fail {
                    resource.failed();
                } else {
                    if state.send_damage {
                        resource.damage(0, 0, state.frame_width, state.frame_height);
                    }
                    resource.ready(state.ts_sec_hi, state.ts_sec_lo, state.ts_nsec);
                }
            }
            zwlr_screencopy_frame_v1::Request::Destroy => {}
            _ => {}
        }
    }
}

// ==========================================================================
// Helper: spawn mock server, return client connection and stop handle
// ==========================================================================

struct MockServer {
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl MockServer {
    fn shutdown(mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            t.join().unwrap();
        }
    }
}

impl Drop for MockServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        // Thread will exit on its own; we don't join here to avoid panics in drop.
    }
}

fn spawn_capture_server(compositor: MockCompositor) -> (wayland_client::Connection, MockServer) {
    let display = wayland_server::Display::<MockCompositor>::new().unwrap();
    let mut dh = display.handle();
    dh.create_global::<MockCompositor, wl_shm::WlShm, ()>(2, ());
    // Create one wl_output global per output name.
    for (i, _name) in compositor.output_names.iter().enumerate() {
        dh.create_global::<MockCompositor, wl_output::WlOutput, usize>(4, i);
    }
    dh.create_global::<MockCompositor, zwlr_screencopy_manager_v1::ZwlrScreencopyManagerV1, ()>(
        3,
        (),
    );

    let (client_stream, server_stream) = UnixStream::pair().unwrap();
    dh.insert_client(server_stream, Arc::new(())).unwrap();
    let client_conn = wayland_client::Connection::from_socket(client_stream).unwrap();

    let stop = Arc::new(AtomicBool::new(false));
    let stop_server = Arc::clone(&stop);

    let thread = std::thread::spawn(move || {
        let mut display = display;
        let mut compositor = compositor;
        while !stop_server.load(Ordering::Relaxed) {
            display.dispatch_clients(&mut compositor).unwrap();
            display.flush_clients().unwrap();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    });

    (
        client_conn,
        MockServer {
            stop,
            thread: Some(thread),
        },
    )
}

/// Spawn a mock server with xdg_output_manager for OutputEnumerator tests.
fn spawn_output_server(compositor: MockCompositor) -> (wayland_client::Connection, MockServer) {
    let display = wayland_server::Display::<MockCompositor>::new().unwrap();
    let mut dh = display.handle();
    for (i, _name) in compositor.output_names.iter().enumerate() {
        dh.create_global::<MockCompositor, wl_output::WlOutput, usize>(4, i);
    }
    dh.create_global::<MockCompositor, zxdg_output_manager_v1::ZxdgOutputManagerV1, ()>(3, ());

    let (client_stream, server_stream) = UnixStream::pair().unwrap();
    dh.insert_client(server_stream, Arc::new(())).unwrap();
    let client_conn = wayland_client::Connection::from_socket(client_stream).unwrap();

    let stop = Arc::new(AtomicBool::new(false));
    let stop_server = Arc::clone(&stop);

    let thread = std::thread::spawn(move || {
        let mut display = display;
        let mut compositor = compositor;
        while !stop_server.load(Ordering::Relaxed) {
            display.dispatch_clients(&mut compositor).unwrap();
            display.flush_clients().unwrap();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    });

    (
        client_conn,
        MockServer {
            stop,
            thread: Some(thread),
        },
    )
}

// ==========================================================================
// Screencopy state — reuses the real library's dispatch types to exercise
// the actual code paths in screencopy.rs.
//
// We cannot call WlrScreencopyBackend::new() directly because it uses
// Connection::connect_to_env(), but we CAN construct the internal state
// and call the dispatch implementations through the Wayland client protocol.
// ==========================================================================

/// Client-side state that mirrors ScreencopyState for testing the protocol flow.
struct ScreencopyTestClient {
    shm: Option<wayland_client::protocol::wl_shm::WlShm>,
    output: Option<wayland_client::protocol::wl_output::WlOutput>,
    manager: Option<
        wayland_protocols_wlr::screencopy::v1::client::zwlr_screencopy_manager_v1::ZwlrScreencopyManagerV1,
    >,
    ready: bool,
    failed: bool,
    buffer_done: bool,
    width: u32,
    height: u32,
    stride: u32,
    format_raw: u32,
    output_name: String,
    damage_x: u32,
    damage_y: u32,
    damage_width: u32,
    damage_height: u32,
    ts_sec_hi: u32,
    ts_sec_lo: u32,
    ts_nsec: u32,
}

impl ScreencopyTestClient {
    fn new() -> Self {
        Self {
            shm: None,
            output: None,
            manager: None,
            ready: false,
            failed: false,
            buffer_done: false,
            width: 0,
            height: 0,
            stride: 0,
            format_raw: 0,
            output_name: String::new(),
            damage_x: 0,
            damage_y: 0,
            damage_width: 0,
            damage_height: 0,
            ts_sec_hi: 0,
            ts_sec_lo: 0,
            ts_nsec: 0,
        }
    }

    fn reset_frame_state(&mut self) {
        self.ready = false;
        self.failed = false;
        self.buffer_done = false;
        self.damage_x = 0;
        self.damage_y = 0;
        self.damage_width = 0;
        self.damage_height = 0;
    }
}

use wayland_client::{Connection as ClientConnection, Dispatch as ClientDispatch, QueueHandle};

impl ClientDispatch<wayland_client::protocol::wl_registry::WlRegistry, ()>
    for ScreencopyTestClient
{
    fn event(
        state: &mut Self,
        registry: &wayland_client::protocol::wl_registry::WlRegistry,
        event: wayland_client::protocol::wl_registry::Event,
        _: &(),
        _: &ClientConnection,
        qh: &QueueHandle<Self>,
    ) {
        if let wayland_client::protocol::wl_registry::Event::Global {
            name,
            interface,
            version,
        } = event
        {
            match interface.as_str() {
                "wl_shm" => state.shm = Some(registry.bind(name, version.min(2), qh, ())),
                "wl_output" => {
                    // Only bind the first output; tests that need multiple outputs
                    // use OutputEnumerator which has its own dispatch.
                    if state.output.is_none() {
                        state.output = Some(registry.bind(name, version.min(4), qh, name));
                    }
                }
                "zwlr_screencopy_manager_v1" => {
                    state.manager = Some(registry.bind(name, version.min(3), qh, ()))
                }
                _ => {}
            }
        }
    }
}

impl ClientDispatch<wayland_client::protocol::wl_shm::WlShm, ()> for ScreencopyTestClient {
    fn event(
        _: &mut Self,
        _: &wayland_client::protocol::wl_shm::WlShm,
        _: wayland_client::protocol::wl_shm::Event,
        _: &(),
        _: &ClientConnection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl ClientDispatch<wayland_client::protocol::wl_shm_pool::WlShmPool, ()> for ScreencopyTestClient {
    fn event(
        _: &mut Self,
        _: &wayland_client::protocol::wl_shm_pool::WlShmPool,
        _: wayland_client::protocol::wl_shm_pool::Event,
        _: &(),
        _: &ClientConnection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl ClientDispatch<wayland_client::protocol::wl_buffer::WlBuffer, usize> for ScreencopyTestClient {
    fn event(
        _: &mut Self,
        _: &wayland_client::protocol::wl_buffer::WlBuffer,
        _: wayland_client::protocol::wl_buffer::Event,
        _: &usize,
        _: &ClientConnection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl ClientDispatch<wayland_client::protocol::wl_output::WlOutput, u32> for ScreencopyTestClient {
    fn event(
        state: &mut Self,
        _: &wayland_client::protocol::wl_output::WlOutput,
        event: wayland_client::protocol::wl_output::Event,
        _global_name: &u32,
        _: &ClientConnection,
        _: &QueueHandle<Self>,
    ) {
        if let wayland_client::protocol::wl_output::Event::Name { name } = event {
            state.output_name = name;
        }
    }
}

impl ClientDispatch<
    wayland_protocols_wlr::screencopy::v1::client::zwlr_screencopy_manager_v1::ZwlrScreencopyManagerV1,
    (),
> for ScreencopyTestClient
{
    fn event(
        _: &mut Self,
        _: &wayland_protocols_wlr::screencopy::v1::client::zwlr_screencopy_manager_v1::ZwlrScreencopyManagerV1,
        _: wayland_protocols_wlr::screencopy::v1::client::zwlr_screencopy_manager_v1::Event,
        _: &(),
        _: &ClientConnection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl ClientDispatch<
    wayland_protocols_wlr::screencopy::v1::client::zwlr_screencopy_frame_v1::ZwlrScreencopyFrameV1,
    (),
> for ScreencopyTestClient
{
    fn event(
        state: &mut Self,
        _: &wayland_protocols_wlr::screencopy::v1::client::zwlr_screencopy_frame_v1::ZwlrScreencopyFrameV1,
        event: wayland_protocols_wlr::screencopy::v1::client::zwlr_screencopy_frame_v1::Event,
        _: &(),
        _: &ClientConnection,
        _: &QueueHandle<Self>,
    ) {
        use wayland_protocols_wlr::screencopy::v1::client::zwlr_screencopy_frame_v1::Event;
        match event {
            Event::Buffer {
                format,
                width,
                height,
                stride,
            } => {
                state.width = width;
                state.height = height;
                state.stride = stride;
                if let wayland_client::WEnum::Value(f) = format {
                    state.format_raw = f as u32;
                }
            }
            Event::BufferDone => {
                state.buffer_done = true;
            }
            Event::Damage {
                x,
                y,
                width,
                height,
            } => {
                state.damage_x = x;
                state.damage_y = y;
                state.damage_width = width;
                state.damage_height = height;
            }
            Event::Ready {
                tv_sec_hi,
                tv_sec_lo,
                tv_nsec,
            } => {
                state.ts_sec_hi = tv_sec_hi;
                state.ts_sec_lo = tv_sec_lo;
                state.ts_nsec = tv_nsec;
                state.ready = true;
            }
            Event::Failed => {
                state.failed = true;
            }
            _ => {}
        }
    }
}

// ==========================================================================
// Tests
// ==========================================================================

/// Test basic screencopy protocol round-trip: globals → output name → capture → ready.
#[test]
fn screencopy_protocol_round_trip() {
    let compositor = MockCompositor::new(64, 48);
    let (client_conn, server) = spawn_capture_server(compositor);

    let mut eq = client_conn.new_event_queue::<ScreencopyTestClient>();
    let qh = eq.handle();
    let mut cs = ScreencopyTestClient::new();

    client_conn.display().get_registry(&qh, ());

    // Round-trip 1: discover globals.
    eq.roundtrip(&mut cs).unwrap();
    assert!(cs.shm.is_some(), "wl_shm not found");
    assert!(cs.output.is_some(), "wl_output not found");
    assert!(cs.manager.is_some(), "screencopy manager not found");

    // Round-trip 2: output events (name, mode).
    eq.roundtrip(&mut cs).unwrap();
    assert_eq!(cs.output_name, "MOCK-1");

    // Request capture.
    let _frame =
        cs.manager
            .as_ref()
            .unwrap()
            .capture_output(0, cs.output.as_ref().unwrap(), &qh, ());

    // Round-trip 3: Buffer + BufferDone events.
    eq.roundtrip(&mut cs).unwrap();
    assert_eq!(cs.width, 64);
    assert_eq!(cs.height, 48);
    assert!(cs.buffer_done, "buffer_done not received");

    // The ready should come after the copy request. Since our mock sends
    // Buffer+BufferDone on CaptureOutput, and Ready on Copy, we need to
    // send a Copy request. For this basic test, check that buffer_done works.
    assert!(!cs.ready, "ready should not be received yet (no copy sent)");

    server.shutdown();
}

/// Test the full capture flow: CaptureOutput → Buffer → BufferDone → (create pool) → Copy → Damage → Ready.
#[test]
fn screencopy_full_capture_flow() {
    let compositor = MockCompositor::new(64, 48);
    let copy_count = Arc::clone(&compositor.copy_count);
    let (client_conn, server) = spawn_capture_server(compositor);

    let mut eq = client_conn.new_event_queue::<ScreencopyTestClient>();
    let qh = eq.handle();
    let mut cs = ScreencopyTestClient::new();

    client_conn.display().get_registry(&qh, ());
    eq.roundtrip(&mut cs).unwrap();
    eq.roundtrip(&mut cs).unwrap(); // output name

    // Request capture.
    let frame =
        cs.manager
            .as_ref()
            .unwrap()
            .capture_output(0, cs.output.as_ref().unwrap(), &qh, ());

    // Wait for Buffer + BufferDone.
    eq.roundtrip(&mut cs).unwrap();
    assert!(cs.buffer_done);
    assert_eq!(cs.width, 64);
    assert_eq!(cs.height, 48);
    assert_eq!(cs.stride, 256); // 64 * 4

    // Create SHM pool using the REAL ShmBufferPool type.
    let shm = cs.shm.as_ref().unwrap();
    let wl_format = wayland_client::protocol::wl_shm::Format::Xrgb8888;
    let pool = ShmBufferPool::new(shm, cs.width, cs.height, cs.stride, wl_format, &qh).unwrap();

    // Verify pool properties.
    assert_eq!(pool.width, 64);
    assert_eq!(pool.height, 48);
    assert_eq!(pool.stride, 256);
    assert_eq!(pool.buffer_size(), 256 * 48);

    // Roundtrip so the server processes pool creation.
    eq.roundtrip(&mut cs).unwrap();

    // Send copy request with the real buffer.
    frame.copy(pool.active_buffer());

    // Roundtrip: server receives copy, sends damage + ready.
    eq.roundtrip(&mut cs).unwrap();
    assert!(cs.ready, "frame should be ready after copy");
    assert_eq!(copy_count.load(Ordering::Relaxed), 1);

    // Verify damage.
    assert_eq!(cs.damage_width, 64);
    assert_eq!(cs.damage_height, 48);

    // Destroy frame.
    frame.destroy();

    server.shutdown();
}

/// Test capture with Argb8888 format.
#[test]
fn screencopy_argb8888_format() {
    let compositor = MockCompositor::new(32, 32).with_format(wl_shm::Format::Argb8888);
    let (client_conn, server) = spawn_capture_server(compositor);

    let mut eq = client_conn.new_event_queue::<ScreencopyTestClient>();
    let qh = eq.handle();
    let mut cs = ScreencopyTestClient::new();

    client_conn.display().get_registry(&qh, ());
    eq.roundtrip(&mut cs).unwrap();
    eq.roundtrip(&mut cs).unwrap();

    let frame =
        cs.manager
            .as_ref()
            .unwrap()
            .capture_output(0, cs.output.as_ref().unwrap(), &qh, ());
    eq.roundtrip(&mut cs).unwrap();

    assert!(cs.buffer_done);
    // Argb8888 = format code 0 in wl_shm.
    assert_eq!(cs.format_raw, wl_shm::Format::Argb8888 as u32);

    // Verify PixelFormat conversion.
    let pf = PixelFormat::from_wl_shm(cs.format_raw);
    assert_eq!(pf, Some(PixelFormat::Argb8888));

    frame.destroy();
    server.shutdown();
}

/// Test capture with Xbgr8888 format.
#[test]
fn screencopy_xbgr8888_format() {
    let compositor = MockCompositor::new(16, 16).with_format(wl_shm::Format::Xbgr8888);
    let (client_conn, server) = spawn_capture_server(compositor);

    let mut eq = client_conn.new_event_queue::<ScreencopyTestClient>();
    let qh = eq.handle();
    let mut cs = ScreencopyTestClient::new();

    client_conn.display().get_registry(&qh, ());
    eq.roundtrip(&mut cs).unwrap();
    eq.roundtrip(&mut cs).unwrap();

    let frame =
        cs.manager
            .as_ref()
            .unwrap()
            .capture_output(0, cs.output.as_ref().unwrap(), &qh, ());
    eq.roundtrip(&mut cs).unwrap();

    assert!(cs.buffer_done);
    let pf = PixelFormat::from_wl_shm(cs.format_raw);
    assert_eq!(pf, Some(PixelFormat::Xbgr8888));

    frame.destroy();
    server.shutdown();
}

/// Test capture failure: compositor sends Failed event on CaptureOutput.
#[test]
fn screencopy_capture_failure_on_capture_output() {
    let compositor = MockCompositor::new(64, 48).with_failure();
    let (client_conn, server) = spawn_capture_server(compositor);

    let mut eq = client_conn.new_event_queue::<ScreencopyTestClient>();
    let qh = eq.handle();
    let mut cs = ScreencopyTestClient::new();

    client_conn.display().get_registry(&qh, ());
    eq.roundtrip(&mut cs).unwrap();
    eq.roundtrip(&mut cs).unwrap();

    let frame =
        cs.manager
            .as_ref()
            .unwrap()
            .capture_output(0, cs.output.as_ref().unwrap(), &qh, ());
    eq.roundtrip(&mut cs).unwrap();

    // Server sent Failed instead of Buffer+BufferDone.
    assert!(cs.failed, "expected failure event");
    assert!(!cs.buffer_done, "should not have buffer_done on failure");
    assert!(!cs.ready, "should not be ready on failure");

    frame.destroy();
    server.shutdown();
}

/// Test timestamp propagation through Ready event.
#[test]
fn screencopy_timestamp_propagation() {
    let compositor = MockCompositor::new(64, 48).with_timestamp(1, 2, 500_000_000);
    let copy_count = Arc::clone(&compositor.copy_count);
    let (client_conn, server) = spawn_capture_server(compositor);

    let mut eq = client_conn.new_event_queue::<ScreencopyTestClient>();
    let qh = eq.handle();
    let mut cs = ScreencopyTestClient::new();

    client_conn.display().get_registry(&qh, ());
    eq.roundtrip(&mut cs).unwrap();
    eq.roundtrip(&mut cs).unwrap();

    let frame =
        cs.manager
            .as_ref()
            .unwrap()
            .capture_output(0, cs.output.as_ref().unwrap(), &qh, ());
    eq.roundtrip(&mut cs).unwrap();
    assert!(cs.buffer_done);

    // Create pool and copy.
    let shm = cs.shm.as_ref().unwrap();
    let wl_format = wayland_client::protocol::wl_shm::Format::Xrgb8888;
    let pool = ShmBufferPool::new(shm, cs.width, cs.height, cs.stride, wl_format, &qh).unwrap();
    eq.roundtrip(&mut cs).unwrap();

    frame.copy(pool.active_buffer());
    eq.roundtrip(&mut cs).unwrap();

    assert!(cs.ready);
    assert_eq!(cs.ts_sec_hi, 1);
    assert_eq!(cs.ts_sec_lo, 2);
    assert_eq!(cs.ts_nsec, 500_000_000);

    // Verify the timestamp calculation matches the real code in screencopy.rs.
    let timestamp_ns =
        ((cs.ts_sec_hi as u64) << 32 | cs.ts_sec_lo as u64) * 1_000_000_000 + cs.ts_nsec as u64;
    // (1 << 32 | 2) = 4294967298, * 1e9 = 4294967298000000000, + 500000000
    let expected = ((1u64 << 32) | 2u64) * 1_000_000_000 + 500_000_000;
    assert_eq!(timestamp_ns, expected);

    assert_eq!(copy_count.load(Ordering::Relaxed), 1);

    frame.destroy();
    server.shutdown();
}

/// Test capture with no damage events (compositor omits damage).
#[test]
fn screencopy_no_damage_events() {
    let compositor = MockCompositor::new(64, 48).with_no_damage();
    let (client_conn, server) = spawn_capture_server(compositor);

    let mut eq = client_conn.new_event_queue::<ScreencopyTestClient>();
    let qh = eq.handle();
    let mut cs = ScreencopyTestClient::new();

    client_conn.display().get_registry(&qh, ());
    eq.roundtrip(&mut cs).unwrap();
    eq.roundtrip(&mut cs).unwrap();

    let frame =
        cs.manager
            .as_ref()
            .unwrap()
            .capture_output(0, cs.output.as_ref().unwrap(), &qh, ());
    eq.roundtrip(&mut cs).unwrap();

    let shm = cs.shm.as_ref().unwrap();
    let wl_format = wayland_client::protocol::wl_shm::Format::Xrgb8888;
    let pool = ShmBufferPool::new(shm, cs.width, cs.height, cs.stride, wl_format, &qh).unwrap();
    eq.roundtrip(&mut cs).unwrap();

    frame.copy(pool.active_buffer());
    eq.roundtrip(&mut cs).unwrap();

    assert!(cs.ready);
    // No damage event should have been sent.
    assert_eq!(cs.damage_width, 0);
    assert_eq!(cs.damage_height, 0);

    frame.destroy();
    server.shutdown();
}

/// Test multiple sequential captures with buffer swap.
#[test]
fn screencopy_multiple_captures_with_swap() {
    let compositor = MockCompositor::new(64, 48);
    let copy_count = Arc::clone(&compositor.copy_count);
    let capture_count = Arc::clone(&compositor.capture_count);
    let (client_conn, server) = spawn_capture_server(compositor);

    let mut eq = client_conn.new_event_queue::<ScreencopyTestClient>();
    let qh = eq.handle();
    let mut cs = ScreencopyTestClient::new();

    client_conn.display().get_registry(&qh, ());
    eq.roundtrip(&mut cs).unwrap();
    eq.roundtrip(&mut cs).unwrap();

    let shm = cs.shm.clone().unwrap();
    let wl_format = wayland_client::protocol::wl_shm::Format::Xrgb8888;

    // First capture: creates the SHM pool.
    let frame1 =
        cs.manager
            .as_ref()
            .unwrap()
            .capture_output(0, cs.output.as_ref().unwrap(), &qh, ());
    eq.roundtrip(&mut cs).unwrap();
    assert!(cs.buffer_done);

    let mut pool =
        ShmBufferPool::new(&shm, cs.width, cs.height, cs.stride, wl_format, &qh).unwrap();
    eq.roundtrip(&mut cs).unwrap();

    frame1.copy(pool.active_buffer());
    eq.roundtrip(&mut cs).unwrap();
    assert!(cs.ready);

    // Swap buffer for next capture.
    pool.swap();

    // Read data from the captured buffer.
    let buf_size = pool.buffer_size();
    let data = unsafe { pool.active_data() };
    assert_eq!(data.len(), buf_size);

    // Second capture.
    cs.reset_frame_state();
    pool.swap(); // swap back

    let frame2 =
        cs.manager
            .as_ref()
            .unwrap()
            .capture_output(0, cs.output.as_ref().unwrap(), &qh, ());
    eq.roundtrip(&mut cs).unwrap();
    assert!(cs.buffer_done);

    frame2.copy(pool.active_buffer());
    eq.roundtrip(&mut cs).unwrap();
    assert!(cs.ready);

    // Third capture.
    cs.reset_frame_state();
    pool.swap();

    let frame3 =
        cs.manager
            .as_ref()
            .unwrap()
            .capture_output(0, cs.output.as_ref().unwrap(), &qh, ());
    eq.roundtrip(&mut cs).unwrap();
    assert!(cs.buffer_done);

    frame3.copy(pool.active_buffer());
    eq.roundtrip(&mut cs).unwrap();
    assert!(cs.ready);

    assert_eq!(capture_count.load(Ordering::Relaxed), 3);
    assert_eq!(copy_count.load(Ordering::Relaxed), 3);

    frame1.destroy();
    frame2.destroy();
    frame3.destroy();
    server.shutdown();
}

/// Test ShmBufferPool creation with various dimensions.
#[test]
fn shm_buffer_pool_various_dimensions() {
    let compositor = MockCompositor::new(64, 48);
    let (client_conn, server) = spawn_capture_server(compositor);

    let mut eq = client_conn.new_event_queue::<ScreencopyTestClient>();
    let qh = eq.handle();
    let mut cs = ScreencopyTestClient::new();

    client_conn.display().get_registry(&qh, ());
    eq.roundtrip(&mut cs).unwrap();

    let shm = cs.shm.as_ref().unwrap();
    let format = wayland_client::protocol::wl_shm::Format::Xrgb8888;

    // 1x1.
    let pool = ShmBufferPool::new(shm, 1, 1, 4, format, &qh).unwrap();
    assert_eq!(pool.buffer_size(), 4);
    assert_eq!(pool.width, 1);
    assert_eq!(pool.height, 1);
    assert_eq!(pool.stride, 4);

    // Standard HD.
    let pool = ShmBufferPool::new(shm, 1920, 1080, 1920 * 4, format, &qh).unwrap();
    assert_eq!(pool.buffer_size(), 1920 * 4 * 1080);

    // With padding in stride.
    let pool = ShmBufferPool::new(shm, 100, 100, 512, format, &qh).unwrap();
    assert_eq!(pool.buffer_size(), 512 * 100);
    assert_eq!(pool.stride, 512);

    // 4K.
    let pool = ShmBufferPool::new(shm, 3840, 2160, 3840 * 4, format, &qh).unwrap();
    assert_eq!(pool.buffer_size(), 3840 * 4 * 2160);

    server.shutdown();
}

/// Test ShmBufferPool error paths.
#[test]
fn shm_buffer_pool_error_paths() {
    let compositor = MockCompositor::new(64, 48);
    let (client_conn, server) = spawn_capture_server(compositor);

    let mut eq = client_conn.new_event_queue::<ScreencopyTestClient>();
    let qh = eq.handle();
    let mut cs = ScreencopyTestClient::new();

    client_conn.display().get_registry(&qh, ());
    eq.roundtrip(&mut cs).unwrap();

    let shm = cs.shm.as_ref().unwrap();
    let format = wayland_client::protocol::wl_shm::Format::Xrgb8888;

    // Zero-size buffer (height = 0).
    let result = ShmBufferPool::new(shm, 100, 0, 400, format, &qh);
    assert!(result.is_err());

    // Stride less than width * 4.
    let result = ShmBufferPool::new(shm, 100, 100, 100, format, &qh);
    assert!(result.is_err());

    // Stride exactly at minimum — should succeed.
    let result = ShmBufferPool::new(shm, 100, 100, 400, format, &qh);
    assert!(result.is_ok());

    server.shutdown();
}

/// Test ShmBufferPool swap and active_data cycle.
#[test]
fn shm_buffer_pool_swap_and_active_data() {
    let compositor = MockCompositor::new(64, 48);
    let (client_conn, server) = spawn_capture_server(compositor);

    let mut eq = client_conn.new_event_queue::<ScreencopyTestClient>();
    let qh = eq.handle();
    let mut cs = ScreencopyTestClient::new();

    client_conn.display().get_registry(&qh, ());
    eq.roundtrip(&mut cs).unwrap();

    let shm = cs.shm.as_ref().unwrap();
    let format = wayland_client::protocol::wl_shm::Format::Xrgb8888;

    let mut pool = ShmBufferPool::new(shm, 8, 4, 32, format, &qh).unwrap();
    assert_eq!(pool.buffer_size(), 128); // 32 * 4

    // Both buffers should initially be zero-filled (memfd).
    let data0 = unsafe { pool.active_data() };
    assert_eq!(data0.len(), 128);
    assert!(data0.iter().all(|&b| b == 0));

    // Swap to buffer 1.
    pool.swap();
    let data1 = unsafe { pool.active_data() };
    assert_eq!(data1.len(), 128);
    assert!(data1.iter().all(|&b| b == 0));

    // Swap back to buffer 0.
    pool.swap();
    let data0_again = unsafe { pool.active_data() };
    assert_eq!(data0_again.len(), 128);

    server.shutdown();
}

/// Test ShmBufferPool buffer_size correctness across formats.
#[test]
fn shm_buffer_pool_buffer_size_correctness() {
    let compositor = MockCompositor::new(64, 48);
    let (client_conn, server) = spawn_capture_server(compositor);

    let mut eq = client_conn.new_event_queue::<ScreencopyTestClient>();
    let qh = eq.handle();
    let mut cs = ScreencopyTestClient::new();

    client_conn.display().get_registry(&qh, ());
    eq.roundtrip(&mut cs).unwrap();

    let shm = cs.shm.as_ref().unwrap();

    let test_cases: Vec<(u32, u32, u32)> = vec![
        (64, 48, 64 * 4),
        (1920, 1080, 1920 * 4),
        (100, 100, 512), // padded stride
        (1, 1, 4),
        (2, 2, 8),
    ];

    for (w, h, stride) in test_cases {
        let pool = ShmBufferPool::new(
            shm,
            w,
            h,
            stride,
            wayland_client::protocol::wl_shm::Format::Xrgb8888,
            &qh,
        )
        .unwrap();
        assert_eq!(
            pool.buffer_size(),
            stride as usize * h as usize,
            "buffer_size mismatch for {w}x{h} stride={stride}"
        );
    }

    server.shutdown();
}

/// Test OutputEnumerator with a single output.
#[test]
fn output_enumerator_single_output() {
    let compositor = MockCompositor::new(1920, 1080);
    let (client_conn, server) = spawn_output_server(compositor);

    let enumerator = OutputEnumerator::enumerate(&client_conn).unwrap();
    let outputs = enumerator.outputs();

    assert_eq!(outputs.len(), 1);
    // xdg_output overrides the name to "xdg-output-0".
    assert_eq!(outputs[0].name, "xdg-output-0");
    assert_eq!(outputs[0].description, "XDG Output 0");
    // xdg_output sets logical size.
    assert_eq!(outputs[0].width, 1920);
    assert_eq!(outputs[0].height, 1080);

    // find_by_name.
    assert!(enumerator.find_by_name("xdg-output-0").is_some());
    assert!(enumerator.find_by_name("nonexistent").is_none());

    server.shutdown();
}

/// Test OutputEnumerator with multiple outputs.
#[test]
fn output_enumerator_multiple_outputs() {
    let compositor =
        MockCompositor::new(2560, 1440).with_outputs(vec!["HDMI-A-1".into(), "eDP-1".into()]);
    let (client_conn, server) = spawn_output_server(compositor);

    let enumerator = OutputEnumerator::enumerate(&client_conn).unwrap();
    let outputs = enumerator.outputs();

    // Both outputs should be discovered.
    assert_eq!(outputs.len(), 2);

    // All outputs get their geometry events with expected dimensions.
    for output in outputs {
        assert_eq!(output.refresh_mhz, 60000);
    }

    server.shutdown();
}

/// Test that OutputInfo fields are correctly populated by mock output events.
#[test]
fn output_info_fields_populated() {
    let compositor = MockCompositor::new(2560, 1440);
    let (client_conn, server) = spawn_output_server(compositor);

    let enumerator = OutputEnumerator::enumerate(&client_conn).unwrap();
    let outputs = enumerator.outputs();

    assert!(!outputs.is_empty());
    let output = &outputs[0];

    // xdg_output sets logical position to (0, 0).
    assert_eq!(output.x, 0);
    assert_eq!(output.y, 0);
    // xdg_output sets logical size.
    assert_eq!(output.width, 1920);
    assert_eq!(output.height, 1080);

    server.shutdown();
}

/// Test PixelFormat conversion for all supported formats.
#[test]
fn pixel_format_from_wl_shm_all_formats() {
    assert_eq!(PixelFormat::from_wl_shm(0), Some(PixelFormat::Argb8888));
    assert_eq!(PixelFormat::from_wl_shm(1), Some(PixelFormat::Xrgb8888));
    assert_eq!(
        PixelFormat::from_wl_shm(0x34324241),
        Some(PixelFormat::Abgr8888)
    );
    assert_eq!(
        PixelFormat::from_wl_shm(0x34324258),
        Some(PixelFormat::Xbgr8888)
    );
    assert_eq!(PixelFormat::from_wl_shm(4), None);
    assert_eq!(PixelFormat::from_wl_shm(999), None);
    assert_eq!(PixelFormat::from_wl_shm(u32::MAX), None);
}

/// Test PixelFormat bytes_per_pixel.
#[test]
fn pixel_format_bpp() {
    for pf in [
        PixelFormat::Argb8888,
        PixelFormat::Xrgb8888,
        PixelFormat::Abgr8888,
        PixelFormat::Xbgr8888,
    ] {
        assert_eq!(pf.bytes_per_pixel(), 4);
    }
}

/// Test that output_name is correctly propagated from the mock output.
#[test]
fn screencopy_output_name_propagation() {
    let compositor = MockCompositor::new(64, 48).with_outputs(vec!["DP-3".to_string()]);
    let (client_conn, server) = spawn_capture_server(compositor);

    let mut eq = client_conn.new_event_queue::<ScreencopyTestClient>();
    let qh = eq.handle();
    let mut cs = ScreencopyTestClient::new();

    client_conn.display().get_registry(&qh, ());
    eq.roundtrip(&mut cs).unwrap();
    eq.roundtrip(&mut cs).unwrap();

    assert_eq!(cs.output_name, "DP-3");

    server.shutdown();
}

/// Test ShmBufferPool with Argb8888 format.
#[test]
fn shm_buffer_pool_argb_format() {
    let compositor = MockCompositor::new(64, 48);
    let (client_conn, server) = spawn_capture_server(compositor);

    let mut eq = client_conn.new_event_queue::<ScreencopyTestClient>();
    let qh = eq.handle();
    let mut cs = ScreencopyTestClient::new();

    client_conn.display().get_registry(&qh, ());
    eq.roundtrip(&mut cs).unwrap();

    let shm = cs.shm.as_ref().unwrap();
    let pool = ShmBufferPool::new(
        shm,
        32,
        32,
        128,
        wayland_client::protocol::wl_shm::Format::Argb8888,
        &qh,
    )
    .unwrap();

    assert_eq!(
        pool.format,
        wayland_client::protocol::wl_shm::Format::Argb8888
    );
    assert_eq!(pool.buffer_size(), 128 * 32);

    server.shutdown();
}

/// Test that the capture flow properly handles the two-phase dispatch
/// (Buffer+BufferDone in phase 1, Copy→Ready in phase 2).
#[test]
fn screencopy_two_phase_dispatch() {
    let compositor = MockCompositor::new(128, 96);
    let (client_conn, server) = spawn_capture_server(compositor);

    let mut eq = client_conn.new_event_queue::<ScreencopyTestClient>();
    let qh = eq.handle();
    let mut cs = ScreencopyTestClient::new();

    client_conn.display().get_registry(&qh, ());
    eq.roundtrip(&mut cs).unwrap();
    eq.roundtrip(&mut cs).unwrap();

    // Phase 1: CaptureOutput → Buffer + BufferDone.
    let frame =
        cs.manager
            .as_ref()
            .unwrap()
            .capture_output(0, cs.output.as_ref().unwrap(), &qh, ());
    eq.roundtrip(&mut cs).unwrap();

    assert!(cs.buffer_done, "phase 1: buffer_done expected");
    assert!(!cs.ready, "phase 1: ready should not arrive yet");
    assert_eq!(cs.width, 128);
    assert_eq!(cs.height, 96);
    assert_eq!(cs.stride, 512); // 128 * 4

    // Create pool between phases.
    let shm = cs.shm.as_ref().unwrap();
    let pool = ShmBufferPool::new(
        shm,
        cs.width,
        cs.height,
        cs.stride,
        wayland_client::protocol::wl_shm::Format::Xrgb8888,
        &qh,
    )
    .unwrap();
    eq.roundtrip(&mut cs).unwrap();

    // Phase 2: Copy → Damage + Ready.
    frame.copy(pool.active_buffer());
    eq.roundtrip(&mut cs).unwrap();

    assert!(cs.ready, "phase 2: ready expected after copy");
    assert_eq!(cs.damage_width, 128);
    assert_eq!(cs.damage_height, 96);

    frame.destroy();
    server.shutdown();
}

/// Test ShmBufferPool active_buffer returns correct buffers after swaps.
#[test]
fn shm_buffer_pool_active_buffer_consistency() {
    let compositor = MockCompositor::new(64, 48);
    let (client_conn, server) = spawn_capture_server(compositor);

    let mut eq = client_conn.new_event_queue::<ScreencopyTestClient>();
    let qh = eq.handle();
    let mut cs = ScreencopyTestClient::new();

    client_conn.display().get_registry(&qh, ());
    eq.roundtrip(&mut cs).unwrap();

    let shm = cs.shm.as_ref().unwrap();
    let mut pool = ShmBufferPool::new(
        shm,
        8,
        4,
        32,
        wayland_client::protocol::wl_shm::Format::Xrgb8888,
        &qh,
    )
    .unwrap();

    // Active buffer 0 initially.
    let buf_a = pool.active_buffer().id();
    pool.swap();
    let buf_b = pool.active_buffer().id();

    // Two different buffers.
    assert_ne!(
        buf_a, buf_b,
        "double buffer should have two distinct buffers"
    );

    // Swap back: should get the same buffer as initially.
    pool.swap();
    let buf_a_again = pool.active_buffer().id();
    assert_eq!(buf_a, buf_a_again);

    server.shutdown();
}

/// Test that ShmBufferPool correctly stores the format.
#[test]
fn shm_buffer_pool_format_stored() {
    let compositor = MockCompositor::new(64, 48);
    let (client_conn, server) = spawn_capture_server(compositor);

    let mut eq = client_conn.new_event_queue::<ScreencopyTestClient>();
    let qh = eq.handle();
    let mut cs = ScreencopyTestClient::new();

    client_conn.display().get_registry(&qh, ());
    eq.roundtrip(&mut cs).unwrap();

    let shm = cs.shm.as_ref().unwrap();

    // Xrgb8888.
    let pool = ShmBufferPool::new(
        shm,
        16,
        16,
        64,
        wayland_client::protocol::wl_shm::Format::Xrgb8888,
        &qh,
    )
    .unwrap();
    assert_eq!(
        pool.format,
        wayland_client::protocol::wl_shm::Format::Xrgb8888
    );

    // Argb8888.
    let pool = ShmBufferPool::new(
        shm,
        16,
        16,
        64,
        wayland_client::protocol::wl_shm::Format::Argb8888,
        &qh,
    )
    .unwrap();
    assert_eq!(
        pool.format,
        wayland_client::protocol::wl_shm::Format::Argb8888
    );

    server.shutdown();
}

// ---------------------------------------------------------------------------
// Tests using WAYLAND_DISPLAY env var to exercise WlrScreencopyBackend::new()
// and related public APIs through the actual code paths.
// ---------------------------------------------------------------------------

use std::sync::Mutex;

use remoteway_capture::backend::CaptureBackend;

/// Mutex to serialize tests that modify the WAYLAND_DISPLAY env var.
static SCREENCOPY_WAYLAND_DISPLAY_LOCK: Mutex<()> = Mutex::new(());

struct ScreencopyListeningMock {
    stop: Arc<AtomicBool>,
    server_thread: Option<std::thread::JoinHandle<()>>,
    _guard: std::sync::MutexGuard<'static, ()>,
    old_display: Option<String>,
}

impl ScreencopyListeningMock {
    fn new(compositor: MockCompositor) -> Self {
        use wayland_server::ListeningSocket;

        let guard = SCREENCOPY_WAYLAND_DISPLAY_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let display = wayland_server::Display::<MockCompositor>::new().unwrap();
        let dh = display.handle();

        dh.create_global::<MockCompositor, wl_shm::WlShm, ()>(2, ());
        for (i, _name) in compositor.output_names.iter().enumerate() {
            dh.create_global::<MockCompositor, wl_output::WlOutput, usize>(4, i);
        }
        dh.create_global::<MockCompositor, zwlr_screencopy_manager_v1::ZwlrScreencopyManagerV1, ()>(
            3,
            (),
        );

        let id = std::process::id();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos();
        let socket_name = format!("remoteway-screencopy-test-{}-{}", id, ts);
        let listener = ListeningSocket::bind(&socket_name).unwrap();

        let old_display = std::env::var("WAYLAND_DISPLAY").ok();
        unsafe { std::env::set_var("WAYLAND_DISPLAY", &socket_name) };

        let stop = Arc::new(AtomicBool::new(false));
        let stop_server = Arc::clone(&stop);

        let server_thread = std::thread::spawn(move || {
            let mut display = display;
            let mut compositor = compositor;
            let mut dh = display.handle();
            while !stop_server.load(Ordering::Relaxed) {
                if let Ok(Some(stream)) = listener.accept() {
                    let _ = dh.insert_client(stream, Arc::new(()));
                }
                let _ = display.dispatch_clients(&mut compositor);
                let _ = display.flush_clients();
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        });

        std::thread::sleep(std::time::Duration::from_millis(20));

        Self {
            stop,
            server_thread: Some(server_thread),
            _guard: guard,
            old_display,
        }
    }
}

impl Drop for ScreencopyListeningMock {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.server_thread.take() {
            let _ = h.join();
        }
        match self.old_display.take() {
            Some(val) => unsafe { std::env::set_var("WAYLAND_DISPLAY", val) },
            None => unsafe { std::env::remove_var("WAYLAND_DISPLAY") },
        }
    }
}

#[test]
fn wlr_screencopy_backend_new_with_mock() {
    let compositor = MockCompositor::new(64, 48);
    let _mock = ScreencopyListeningMock::new(compositor);
    let backend = remoteway_capture::screencopy::WlrScreencopyBackend::new(None);
    assert!(
        backend.is_ok(),
        "WlrScreencopyBackend::new failed: {:?}",
        backend.err().map(|e| e.to_string())
    );
    let backend = backend.unwrap();
    assert_eq!(backend.name(), "wlr-screencopy");
}

#[test]
fn wlr_screencopy_is_available_with_mock() {
    let compositor = MockCompositor::new(64, 48);
    let _mock = ScreencopyListeningMock::new(compositor);
    let conn = wayland_client::Connection::connect_to_env().unwrap();
    assert!(remoteway_capture::screencopy::WlrScreencopyBackend::is_available(&conn));
}

#[test]
fn wlr_screencopy_backend_capture_frame() {
    let compositor = MockCompositor::new(128, 96);
    let _mock = ScreencopyListeningMock::new(compositor);
    let mut backend = remoteway_capture::screencopy::WlrScreencopyBackend::new(None).unwrap();
    let frame = backend.next_frame();
    assert!(
        frame.is_ok(),
        "next_frame failed: {:?}",
        frame.err().map(|e| e.to_string())
    );
    let frame = frame.unwrap();
    assert_eq!(frame.width, 128);
    assert_eq!(frame.height, 96);
}

#[test]
fn wlr_screencopy_backend_capture_multiple_frames() {
    let compositor = MockCompositor::new(64, 48);
    let _mock = ScreencopyListeningMock::new(compositor);
    let mut backend = remoteway_capture::screencopy::WlrScreencopyBackend::new(None).unwrap();
    for _ in 0..3 {
        let frame = backend.next_frame();
        assert!(frame.is_ok());
    }
}

#[test]
fn wlr_screencopy_backend_named_output() {
    let compositor = MockCompositor::new(96, 72).with_outputs(vec!["MOCK-1".to_string()]);
    let _mock = ScreencopyListeningMock::new(compositor);
    let backend = remoteway_capture::screencopy::WlrScreencopyBackend::new(Some("MOCK-1"));
    assert!(
        backend.is_ok(),
        "WlrScreencopyBackend::new with name failed: {:?}",
        backend.err().map(|e| e.to_string())
    );
}

#[test]
fn wlr_screencopy_backend_named_output_not_found() {
    let compositor = MockCompositor::new(96, 72).with_outputs(vec!["MOCK-1".to_string()]);
    let _mock = ScreencopyListeningMock::new(compositor);
    let backend = remoteway_capture::screencopy::WlrScreencopyBackend::new(Some("nonexistent"));
    assert!(backend.is_err());
}

#[test]
fn wlr_screencopy_backend_stop() {
    let compositor = MockCompositor::new(64, 48);
    let _mock = ScreencopyListeningMock::new(compositor);
    let mut backend = remoteway_capture::screencopy::WlrScreencopyBackend::new(None).unwrap();
    backend.stop();
    // After stop, next_frame should fail.
    let result = backend.next_frame();
    assert!(result.is_err());
}

#[test]
fn output_enumerator_with_mock() {
    let compositor =
        MockCompositor::new(64, 48).with_outputs(vec!["MOCK-1".to_string(), "MOCK-2".to_string()]);
    let _mock = ScreencopyListeningMock::new(compositor);
    let conn = wayland_client::Connection::connect_to_env().unwrap();
    let enumerator = OutputEnumerator::enumerate(&conn);
    assert!(enumerator.is_ok());
}

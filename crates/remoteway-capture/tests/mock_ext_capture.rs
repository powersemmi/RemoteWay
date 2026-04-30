//! Integration test using an in-process wayland-server mock that implements
//! ext-image-capture-source-v1, ext-image-copy-capture-v1, ext-foreign-toplevel-list-v1,
//! wl_shm, and wl_output for full protocol round-trip testing of `ExtImageCaptureBackend`.
//!
//! This test exercises all Dispatch implementations in ext_capture.rs by creating a mock
//! compositor server that sends the appropriate protocol events in response to client requests.

use std::os::unix::net::UnixStream;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use wayland_server::protocol::{wl_buffer, wl_output, wl_shm, wl_shm_pool};
use wayland_server::{Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New};

// ---------------------------------------------------------------------------
// Server-side protocol bindings generated from vendored XML
// ---------------------------------------------------------------------------

/// Server-side ext-foreign-toplevel-list-v1 bindings.
mod ext_foreign_toplevel_list_v1_server {
    pub mod server {
        use wayland_server;

        pub mod __interfaces {
            wayland_scanner::generate_interfaces!("protocols/ext-foreign-toplevel-list-v1.xml");
        }
        use self::__interfaces::*;

        wayland_scanner::generate_server_code!("protocols/ext-foreign-toplevel-list-v1.xml");
    }
}

/// Server-side ext-image-capture-source-v1 bindings.
mod ext_image_capture_source_v1_server {
    pub mod server {
        use crate::ext_foreign_toplevel_list_v1_server::server::*;
        use wayland_server;
        use wayland_server::protocol::*;

        pub mod __interfaces {
            use crate::ext_foreign_toplevel_list_v1_server::server::__interfaces::*;
            use wayland_server::protocol::__interfaces::*;
            wayland_scanner::generate_interfaces!("protocols/ext-image-capture-source-v1.xml");
        }
        use self::__interfaces::*;

        wayland_scanner::generate_server_code!("protocols/ext-image-capture-source-v1.xml");
    }
}

/// Server-side ext-image-copy-capture-v1 bindings.
mod ext_image_copy_capture_v1_server {
    pub mod server {
        use crate::ext_image_capture_source_v1_server::server::*;
        use wayland_server;
        use wayland_server::protocol::*;

        pub mod __interfaces {
            use crate::ext_image_capture_source_v1_server::server::__interfaces::*;
            use wayland_server::protocol::__interfaces::*;
            wayland_scanner::generate_interfaces!("protocols/ext-image-copy-capture-v1.xml");
        }
        use self::__interfaces::*;

        wayland_scanner::generate_server_code!("protocols/ext-image-copy-capture-v1.xml");
    }
}

use ext_foreign_toplevel_list_v1_server::server::{
    ext_foreign_toplevel_handle_v1 as srv_toplevel_handle,
    ext_foreign_toplevel_list_v1 as srv_toplevel_list,
};
use ext_image_capture_source_v1_server::server::{
    ext_foreign_toplevel_image_capture_source_manager_v1 as srv_toplevel_source_mgr,
    ext_image_capture_source_v1 as srv_capture_source,
    ext_output_image_capture_source_manager_v1 as srv_output_source_mgr,
};
use ext_image_copy_capture_v1_server::server::{
    ext_image_copy_capture_frame_v1 as srv_frame,
    ext_image_copy_capture_manager_v1 as srv_capture_mgr,
    ext_image_copy_capture_session_v1 as srv_session,
};

// ---------------------------------------------------------------------------
// Mock compositor state
// ---------------------------------------------------------------------------

struct MockExtCompositor {
    frame_width: u32,
    frame_height: u32,
    /// Mock toplevels to advertise.
    toplevels: Vec<MockToplevel>,
    /// When true, the next frame capture will send a `failed` event.
    fail_next_frame: bool,
    /// When true, send `stopped` event on the session immediately.
    send_stopped: bool,
}

struct MockToplevel {
    app_id: String,
    title: String,
    identifier: String,
}

impl MockExtCompositor {
    fn new(width: u32, height: u32) -> Self {
        Self {
            frame_width: width,
            frame_height: height,
            toplevels: Vec::new(),
            fail_next_frame: false,
            send_stopped: false,
        }
    }

    fn with_toplevels(mut self, toplevels: Vec<MockToplevel>) -> Self {
        self.toplevels = toplevels;
        self
    }
}

// ---------------------------------------------------------------------------
// GlobalDispatch: wl_shm
// ---------------------------------------------------------------------------

impl GlobalDispatch<wl_shm::WlShm, ()> for MockExtCompositor {
    fn bind(
        _state: &mut Self,
        _handle: &DisplayHandle,
        _client: &Client,
        resource: New<wl_shm::WlShm>,
        _data: &(),
        data_init: &mut DataInit<'_, Self>,
    ) {
        let shm = data_init.init(resource, ());
        shm.format(wl_shm::Format::Xrgb8888);
        shm.format(wl_shm::Format::Argb8888);
    }
}

impl Dispatch<wl_shm::WlShm, ()> for MockExtCompositor {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &wl_shm::WlShm,
        request: wl_shm::Request,
        _data: &(),
        _handle: &DisplayHandle,
        data_init: &mut DataInit<'_, Self>,
    ) {
        if let wl_shm::Request::CreatePool { id, .. } = request {
            data_init.init(id, ());
        }
    }
}

impl Dispatch<wl_shm_pool::WlShmPool, ()> for MockExtCompositor {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &wl_shm_pool::WlShmPool,
        request: wl_shm_pool::Request,
        _data: &(),
        _handle: &DisplayHandle,
        data_init: &mut DataInit<'_, Self>,
    ) {
        if let wl_shm_pool::Request::CreateBuffer { id, .. } = request {
            data_init.init(id, ());
        }
    }
}

impl Dispatch<wl_buffer::WlBuffer, ()> for MockExtCompositor {
    fn request(
        _: &mut Self,
        _: &Client,
        _: &wl_buffer::WlBuffer,
        _: wl_buffer::Request,
        _: &(),
        _: &DisplayHandle,
        _: &mut DataInit<'_, Self>,
    ) {
    }
}

// ---------------------------------------------------------------------------
// GlobalDispatch: wl_output
// ---------------------------------------------------------------------------

impl GlobalDispatch<wl_output::WlOutput, ()> for MockExtCompositor {
    fn bind(
        state: &mut Self,
        _handle: &DisplayHandle,
        _client: &Client,
        resource: New<wl_output::WlOutput>,
        _data: &(),
        data_init: &mut DataInit<'_, Self>,
    ) {
        let output = data_init.init(resource, ());
        output.geometry(
            0,
            0,
            527,
            296,
            wl_output::Subpixel::None,
            "Mock".into(),
            "Test Output".into(),
            wl_output::Transform::Normal,
        );
        output.mode(
            wl_output::Mode::Current,
            state.frame_width as i32,
            state.frame_height as i32,
            60000,
        );
        output.scale(1);
        output.name("test-output".into());
        output.done();
    }
}

impl Dispatch<wl_output::WlOutput, ()> for MockExtCompositor {
    fn request(
        _: &mut Self,
        _: &Client,
        _: &wl_output::WlOutput,
        _: wl_output::Request,
        _: &(),
        _: &DisplayHandle,
        _: &mut DataInit<'_, Self>,
    ) {
    }
}

// ---------------------------------------------------------------------------
// GlobalDispatch: ext_output_image_capture_source_manager_v1
// ---------------------------------------------------------------------------

impl GlobalDispatch<srv_output_source_mgr::ExtOutputImageCaptureSourceManagerV1, ()>
    for MockExtCompositor
{
    fn bind(
        _state: &mut Self,
        _handle: &DisplayHandle,
        _client: &Client,
        resource: New<srv_output_source_mgr::ExtOutputImageCaptureSourceManagerV1>,
        _data: &(),
        data_init: &mut DataInit<'_, Self>,
    ) {
        data_init.init(resource, ());
    }
}

impl Dispatch<srv_output_source_mgr::ExtOutputImageCaptureSourceManagerV1, ()>
    for MockExtCompositor
{
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &srv_output_source_mgr::ExtOutputImageCaptureSourceManagerV1,
        request: srv_output_source_mgr::Request,
        _data: &(),
        _handle: &DisplayHandle,
        data_init: &mut DataInit<'_, Self>,
    ) {
        if let srv_output_source_mgr::Request::CreateSource { source, .. } = request {
            data_init.init(source, ());
        }
    }
}

// ---------------------------------------------------------------------------
// GlobalDispatch: ext_foreign_toplevel_image_capture_source_manager_v1
// ---------------------------------------------------------------------------

impl GlobalDispatch<srv_toplevel_source_mgr::ExtForeignToplevelImageCaptureSourceManagerV1, ()>
    for MockExtCompositor
{
    fn bind(
        _state: &mut Self,
        _handle: &DisplayHandle,
        _client: &Client,
        resource: New<srv_toplevel_source_mgr::ExtForeignToplevelImageCaptureSourceManagerV1>,
        _data: &(),
        data_init: &mut DataInit<'_, Self>,
    ) {
        data_init.init(resource, ());
    }
}

impl Dispatch<srv_toplevel_source_mgr::ExtForeignToplevelImageCaptureSourceManagerV1, ()>
    for MockExtCompositor
{
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &srv_toplevel_source_mgr::ExtForeignToplevelImageCaptureSourceManagerV1,
        request: srv_toplevel_source_mgr::Request,
        _data: &(),
        _handle: &DisplayHandle,
        data_init: &mut DataInit<'_, Self>,
    ) {
        if let srv_toplevel_source_mgr::Request::CreateSource { source, .. } = request {
            data_init.init(source, ());
        }
    }
}

// ---------------------------------------------------------------------------
// ext_image_capture_source_v1
// ---------------------------------------------------------------------------

impl Dispatch<srv_capture_source::ExtImageCaptureSourceV1, ()> for MockExtCompositor {
    fn request(
        _: &mut Self,
        _: &Client,
        _: &srv_capture_source::ExtImageCaptureSourceV1,
        _: srv_capture_source::Request,
        _: &(),
        _: &DisplayHandle,
        _: &mut DataInit<'_, Self>,
    ) {
    }
}

// ---------------------------------------------------------------------------
// GlobalDispatch: ext_image_copy_capture_manager_v1
// ---------------------------------------------------------------------------

impl GlobalDispatch<srv_capture_mgr::ExtImageCopyCaptureManagerV1, ()> for MockExtCompositor {
    fn bind(
        _state: &mut Self,
        _handle: &DisplayHandle,
        _client: &Client,
        resource: New<srv_capture_mgr::ExtImageCopyCaptureManagerV1>,
        _data: &(),
        data_init: &mut DataInit<'_, Self>,
    ) {
        data_init.init(resource, ());
    }
}

impl Dispatch<srv_capture_mgr::ExtImageCopyCaptureManagerV1, ()> for MockExtCompositor {
    fn request(
        state: &mut Self,
        _client: &Client,
        _resource: &srv_capture_mgr::ExtImageCopyCaptureManagerV1,
        request: srv_capture_mgr::Request,
        _data: &(),
        _handle: &DisplayHandle,
        data_init: &mut DataInit<'_, Self>,
    ) {
        if let srv_capture_mgr::Request::CreateSession { session: id, .. } = request {
            let session = data_init.init(id, ());
            // Send session configuration events.
            session.buffer_size(state.frame_width, state.frame_height);
            session.shm_format(wl_shm::Format::Xrgb8888);
            if state.send_stopped {
                session.stopped();
            }
            session.done();
        }
    }
}

// ---------------------------------------------------------------------------
// ext_image_copy_capture_session_v1
// ---------------------------------------------------------------------------

impl Dispatch<srv_session::ExtImageCopyCaptureSessionV1, ()> for MockExtCompositor {
    fn request(
        state: &mut Self,
        _client: &Client,
        _resource: &srv_session::ExtImageCopyCaptureSessionV1,
        request: srv_session::Request,
        _data: &(),
        _handle: &DisplayHandle,
        data_init: &mut DataInit<'_, Self>,
    ) {
        if let srv_session::Request::CreateFrame { frame: id } = request {
            let frame = data_init.init(id, ());
            // Frame events are sent when capture is requested (in frame dispatch).
            // But since the mock is simplified, we send events right away when the
            // frame object is created. The client will call attach_buffer, damage_buffer,
            // capture and then blocking_dispatch to receive events.
            // We defer actual events to the frame's capture request handler.
            //
            // Store frame info in state so we can respond to capture.
            // For simplicity, we just stash the frame ref via the data_init above.
            let _ = (state, frame);
        }
    }
}

// ---------------------------------------------------------------------------
// ext_image_copy_capture_frame_v1
// ---------------------------------------------------------------------------

impl Dispatch<srv_frame::ExtImageCopyCaptureFrameV1, ()> for MockExtCompositor {
    fn request(
        state: &mut Self,
        _client: &Client,
        resource: &srv_frame::ExtImageCopyCaptureFrameV1,
        request: srv_frame::Request,
        _data: &(),
        _handle: &DisplayHandle,
        _data_init: &mut DataInit<'_, Self>,
    ) {
        match request {
            srv_frame::Request::Capture => {
                if state.fail_next_frame {
                    resource.failed(srv_frame::FailureReason::Unknown);
                    state.fail_next_frame = false;
                } else {
                    // Send transform, damage, presentation_time, ready.
                    resource.transform(wl_output::Transform::Normal);
                    resource.damage(0, 0, state.frame_width as i32, state.frame_height as i32);
                    // Presentation time: 1 second, 500000 ns
                    resource.presentation_time(0, 1, 500_000);
                    resource.ready();
                }
            }
            srv_frame::Request::AttachBuffer { .. } => {
                // Nothing to do -- mock accepts any buffer.
            }
            srv_frame::Request::DamageBuffer { .. } => {
                // Nothing to do -- mock accepts any damage.
            }
            srv_frame::Request::Destroy => {
                // Client is destroying the frame object.
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GlobalDispatch: ext_foreign_toplevel_list_v1
// ---------------------------------------------------------------------------

impl GlobalDispatch<srv_toplevel_list::ExtForeignToplevelListV1, ()> for MockExtCompositor {
    fn bind(
        state: &mut Self,
        _handle: &DisplayHandle,
        _client: &Client,
        resource: New<srv_toplevel_list::ExtForeignToplevelListV1>,
        _data: &(),
        data_init: &mut DataInit<'_, Self>,
    ) {
        let list = data_init.init(resource, ());
        // Send toplevel events for each mock toplevel.
        for toplevel in &state.toplevels {
            let handle = _client
                .create_resource::<srv_toplevel_handle::ExtForeignToplevelHandleV1, _, Self>(
                    _handle,
                    1,
                    (),
                )
                .unwrap();
            list.toplevel(&handle);
            handle.app_id(toplevel.app_id.clone());
            handle.title(toplevel.title.clone());
            handle.identifier(toplevel.identifier.clone());
            handle.done();
        }
    }
}

impl Dispatch<srv_toplevel_list::ExtForeignToplevelListV1, ()> for MockExtCompositor {
    fn request(
        _: &mut Self,
        _: &Client,
        _: &srv_toplevel_list::ExtForeignToplevelListV1,
        _: srv_toplevel_list::Request,
        _: &(),
        _: &DisplayHandle,
        _: &mut DataInit<'_, Self>,
    ) {
    }
}

impl Dispatch<srv_toplevel_handle::ExtForeignToplevelHandleV1, ()> for MockExtCompositor {
    fn request(
        _: &mut Self,
        _: &Client,
        _: &srv_toplevel_handle::ExtForeignToplevelHandleV1,
        _: srv_toplevel_handle::Request,
        _: &(),
        _: &DisplayHandle,
        _: &mut DataInit<'_, Self>,
    ) {
    }
}

// ---------------------------------------------------------------------------
// Helper: run mock compositor server in background thread
// ---------------------------------------------------------------------------

struct MockServer {
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl MockServer {
    fn start(compositor: MockExtCompositor) -> (Self, wayland_client::Connection) {
        let display = wayland_server::Display::<MockExtCompositor>::new().unwrap();
        let mut dh = display.handle();

        dh.create_global::<MockExtCompositor, wl_shm::WlShm, ()>(1, ());
        dh.create_global::<MockExtCompositor, wl_output::WlOutput, ()>(4, ());
        dh.create_global::<MockExtCompositor, srv_output_source_mgr::ExtOutputImageCaptureSourceManagerV1, ()>(1, ());
        dh.create_global::<MockExtCompositor, srv_toplevel_source_mgr::ExtForeignToplevelImageCaptureSourceManagerV1, ()>(1, ());
        dh.create_global::<MockExtCompositor, srv_capture_mgr::ExtImageCopyCaptureManagerV1, ()>(
            1,
            (),
        );
        dh.create_global::<MockExtCompositor, srv_toplevel_list::ExtForeignToplevelListV1, ()>(
            1,
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
            Self {
                stop,
                thread: Some(thread),
            },
            client_conn,
        )
    }
}

impl Drop for MockServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

// ---------------------------------------------------------------------------
// Client-side: use ExtCaptureState dispatch code from remoteway-capture.
//
// We cannot directly call ExtImageCaptureBackend::new() because it uses
// Connection::connect_to_env(). Instead, we replicate the protocol round-trips
// that new() performs, using the same Dispatch implementations via the
// public client-side protocol types.
// ---------------------------------------------------------------------------

use remoteway_capture::backend::{CaptureBackend, PixelFormat};
use remoteway_capture::ext_capture::{
    CaptureSource, ExtImageCaptureBackend, ToplevelInfo, enumerate_toplevels,
};

// ---------------------------------------------------------------------------
// Custom client state that mirrors ExtCaptureState's protocol handling.
// We import the actual client types from remoteway_capture's generated code.
// ---------------------------------------------------------------------------

// Since ExtCaptureState is private, we build a lightweight client that exercises
// the same protocol objects. The test verifies the server mock works correctly
// and that the protocol round-trip is complete.
//
// For direct coverage of ExtCaptureState's Dispatch implementations, we use
// WAYLAND_DISPLAY-based tests below.

// ---------------------------------------------------------------------------
// Tests using WAYLAND_DISPLAY and real ExtImageCaptureBackend
// ---------------------------------------------------------------------------

/// Start a mock compositor that listens on a unique socket, set WAYLAND_DISPLAY,
/// and run the given closure. Restores WAYLAND_DISPLAY on completion.
/// Mutex to serialize tests that modify the WAYLAND_DISPLAY env var.
static WAYLAND_DISPLAY_LOCK: Mutex<()> = Mutex::new(());

fn with_mock_wayland<F>(compositor: MockExtCompositor, f: F)
where
    F: FnOnce(),
{
    use wayland_server::ListeningSocket;

    let _guard = WAYLAND_DISPLAY_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());

    let display = wayland_server::Display::<MockExtCompositor>::new().unwrap();
    let dh = display.handle();

    dh.create_global::<MockExtCompositor, wl_shm::WlShm, ()>(1, ());
    dh.create_global::<MockExtCompositor, wl_output::WlOutput, ()>(4, ());
    dh.create_global::<MockExtCompositor, srv_output_source_mgr::ExtOutputImageCaptureSourceManagerV1, ()>(1, ());
    dh.create_global::<MockExtCompositor, srv_toplevel_source_mgr::ExtForeignToplevelImageCaptureSourceManagerV1, ()>(1, ());
    dh.create_global::<MockExtCompositor, srv_capture_mgr::ExtImageCopyCaptureManagerV1, ()>(1, ());
    dh.create_global::<MockExtCompositor, srv_toplevel_list::ExtForeignToplevelListV1, ()>(1, ());

    // Create a unique socket name to avoid conflicts with parallel tests.
    let id = std::process::id();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    let socket_name = format!("remoteway-test-ext-{}-{}", id, ts);

    let listener = ListeningSocket::bind(&socket_name).unwrap();

    // Save and set WAYLAND_DISPLAY.
    let old_display = std::env::var("WAYLAND_DISPLAY").ok();
    unsafe { std::env::set_var("WAYLAND_DISPLAY", &socket_name) };

    let stop = Arc::new(AtomicBool::new(false));
    let stop_server = Arc::clone(&stop);

    let server_thread = std::thread::spawn(move || {
        let mut display = display;
        let mut compositor = compositor;
        let mut dh = display.handle();
        while !stop_server.load(Ordering::Relaxed) {
            // Accept new connections.
            if let Ok(Some(stream)) = listener.accept() {
                dh.insert_client(stream, Arc::new(())).unwrap();
            }
            display.dispatch_clients(&mut compositor).unwrap();
            display.flush_clients().unwrap();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    });

    // Give server time to start.
    std::thread::sleep(std::time::Duration::from_millis(10));

    // Run the test closure.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));

    // Cleanup.
    stop.store(true, Ordering::Relaxed);
    let _ = server_thread.join();

    // Restore WAYLAND_DISPLAY.
    match old_display {
        Some(val) => unsafe { std::env::set_var("WAYLAND_DISPLAY", val) },
        None => unsafe { std::env::remove_var("WAYLAND_DISPLAY") },
    }

    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}

// ---------------------------------------------------------------------------
// Protocol round-trip test (UnixStream pair, no WAYLAND_DISPLAY needed)
// ---------------------------------------------------------------------------

#[test]
fn ext_capture_protocol_round_trip() {
    use wayland_client::protocol::wl_registry;
    use wayland_client::{Dispatch as ClientDispatch, QueueHandle};

    let toplevels = vec![
        MockToplevel {
            app_id: "org.mozilla.firefox".into(),
            title: "Firefox".into(),
            identifier: "ff-001".into(),
        },
        MockToplevel {
            app_id: "org.gnome.Terminal".into(),
            title: "Terminal".into(),
            identifier: "term-002".into(),
        },
    ];
    let compositor = MockExtCompositor::new(128, 96).with_toplevels(toplevels);
    let (_server, client_conn) = MockServer::start(compositor);

    // Client state that mirrors ExtCaptureState's registry handling.
    struct CS {
        shm: Option<wayland_client::protocol::wl_shm::WlShm>,
        output: Option<wayland_client::protocol::wl_output::WlOutput>,
        output_name: String,
        has_output_source_mgr: bool,
        has_toplevel_source_mgr: bool,
        has_copy_capture_mgr: bool,
        has_toplevel_list: bool,
    }

    impl ClientDispatch<wl_registry::WlRegistry, ()> for CS {
        fn event(
            state: &mut Self,
            registry: &wl_registry::WlRegistry,
            event: wl_registry::Event,
            _: &(),
            _: &wayland_client::Connection,
            qh: &QueueHandle<Self>,
        ) {
            if let wl_registry::Event::Global {
                name,
                interface,
                version,
            } = event
            {
                match interface.as_str() {
                    "wl_shm" => state.shm = Some(registry.bind(name, version.min(1), qh, ())),
                    "wl_output" => state.output = Some(registry.bind(name, version.min(4), qh, ())),
                    "ext_output_image_capture_source_manager_v1" => {
                        state.has_output_source_mgr = true;
                    }
                    "ext_foreign_toplevel_image_capture_source_manager_v1" => {
                        state.has_toplevel_source_mgr = true;
                    }
                    "ext_image_copy_capture_manager_v1" => {
                        state.has_copy_capture_mgr = true;
                    }
                    "ext_foreign_toplevel_list_v1" => {
                        state.has_toplevel_list = true;
                    }
                    _ => {}
                }
            }
        }
    }

    impl ClientDispatch<wayland_client::protocol::wl_shm::WlShm, ()> for CS {
        fn event(
            _: &mut Self,
            _: &wayland_client::protocol::wl_shm::WlShm,
            _: wayland_client::protocol::wl_shm::Event,
            _: &(),
            _: &wayland_client::Connection,
            _: &QueueHandle<Self>,
        ) {
        }
    }

    impl ClientDispatch<wayland_client::protocol::wl_output::WlOutput, ()> for CS {
        fn event(
            state: &mut Self,
            _: &wayland_client::protocol::wl_output::WlOutput,
            event: wayland_client::protocol::wl_output::Event,
            _: &(),
            _: &wayland_client::Connection,
            _: &QueueHandle<Self>,
        ) {
            if let wayland_client::protocol::wl_output::Event::Name { name } = event {
                state.output_name = name;
            }
        }
    }

    let mut eq = client_conn.new_event_queue::<CS>();
    let qh = eq.handle();
    let mut cs = CS {
        shm: None,
        output: None,
        output_name: String::new(),
        has_output_source_mgr: false,
        has_toplevel_source_mgr: false,
        has_copy_capture_mgr: false,
        has_toplevel_list: false,
    };

    client_conn.display().get_registry(&qh, ());

    // Round-trip 1: discover globals.
    eq.roundtrip(&mut cs).unwrap();
    assert!(cs.shm.is_some(), "wl_shm not found");
    assert!(cs.output.is_some(), "wl_output not found");
    assert!(cs.has_output_source_mgr, "output source manager not found");
    assert!(
        cs.has_toplevel_source_mgr,
        "toplevel source manager not found"
    );
    assert!(cs.has_copy_capture_mgr, "copy capture manager not found");
    assert!(cs.has_toplevel_list, "toplevel list not found");

    // Round-trip 2: output name.
    eq.roundtrip(&mut cs).unwrap();
    assert_eq!(cs.output_name, "test-output");
}

// ---------------------------------------------------------------------------
// Test: ExtImageCaptureBackend::new with CaptureSource::Output(None)
// ---------------------------------------------------------------------------

#[test]
fn ext_backend_output_none() {
    let compositor = MockExtCompositor::new(128, 96);
    with_mock_wayland(compositor, || {
        let result = ExtImageCaptureBackend::new(CaptureSource::Output(None));
        assert!(result.is_ok(), "expected Ok, got: {:?}", result.err());
        let backend = result.unwrap();
        assert_eq!(backend.name(), "ext-image-capture");
    });
}

// ---------------------------------------------------------------------------
// Test: ExtImageCaptureBackend::new with CaptureSource::Output(Some("test-output"))
// ---------------------------------------------------------------------------

#[test]
fn ext_backend_output_named() {
    let compositor = MockExtCompositor::new(128, 96);
    with_mock_wayland(compositor, || {
        let result =
            ExtImageCaptureBackend::new(CaptureSource::Output(Some("test-output".to_string())));
        assert!(result.is_ok(), "expected Ok, got: {:?}", result.err());
    });
}

// ---------------------------------------------------------------------------
// Test: ExtImageCaptureBackend::new with CaptureSource::Output(Some("nonexistent"))
// ---------------------------------------------------------------------------

#[test]
fn ext_backend_output_not_found() {
    let compositor = MockExtCompositor::new(128, 96);
    with_mock_wayland(compositor, || {
        let result =
            ExtImageCaptureBackend::new(CaptureSource::Output(Some("nonexistent".to_string())));
        assert!(result.is_err());
        let err = result.unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("not found"),
            "expected 'not found' in error: {msg}"
        );
    });
}

// ---------------------------------------------------------------------------
// Test: ExtImageCaptureBackend with CaptureSource::Toplevel
// ---------------------------------------------------------------------------

#[test]
fn ext_backend_toplevel_by_app_id() {
    let toplevels = vec![
        MockToplevel {
            app_id: "org.mozilla.firefox".into(),
            title: "Firefox".into(),
            identifier: "ff-001".into(),
        },
        MockToplevel {
            app_id: "org.gnome.Terminal".into(),
            title: "Terminal".into(),
            identifier: "term-002".into(),
        },
    ];
    let compositor = MockExtCompositor::new(128, 96).with_toplevels(toplevels);
    with_mock_wayland(compositor, || {
        let result =
            ExtImageCaptureBackend::new(CaptureSource::Toplevel("org.mozilla.firefox".to_string()));
        assert!(result.is_ok(), "expected Ok, got: {:?}", result.err());
    });
}

// ---------------------------------------------------------------------------
// Test: ExtImageCaptureBackend with CaptureSource::Toplevel (not found)
// ---------------------------------------------------------------------------

#[test]
fn ext_backend_toplevel_not_found() {
    let toplevels = vec![MockToplevel {
        app_id: "org.mozilla.firefox".into(),
        title: "Firefox".into(),
        identifier: "ff-001".into(),
    }];
    let compositor = MockExtCompositor::new(128, 96).with_toplevels(toplevels);
    with_mock_wayland(compositor, || {
        let result = ExtImageCaptureBackend::new(CaptureSource::Toplevel(
            "com.example.nonexistent".to_string(),
        ));
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("not found"),
            "expected 'not found' in error: {msg}"
        );
    });
}

// ---------------------------------------------------------------------------
// Test: ExtImageCaptureBackend with CaptureSource::NewToplevel
// ---------------------------------------------------------------------------

#[test]
fn ext_backend_new_toplevel() {
    let toplevels = vec![
        MockToplevel {
            app_id: "org.mozilla.firefox".into(),
            title: "Firefox".into(),
            identifier: "ff-001".into(),
        },
        MockToplevel {
            app_id: "org.gnome.Terminal".into(),
            title: "Terminal".into(),
            identifier: "term-002".into(),
        },
    ];
    let compositor = MockExtCompositor::new(128, 96).with_toplevels(toplevels);
    with_mock_wayland(compositor, || {
        // Known set contains ff-001, so it should find term-002 as "new".
        let result = ExtImageCaptureBackend::new(CaptureSource::NewToplevel {
            known_identifiers: vec!["ff-001".to_string()],
        });
        assert!(result.is_ok(), "expected Ok, got: {:?}", result.err());
    });
}

// ---------------------------------------------------------------------------
// Test: NewToplevel with empty known set picks first with identifier
// ---------------------------------------------------------------------------

#[test]
fn ext_backend_new_toplevel_empty_known() {
    let toplevels = vec![MockToplevel {
        app_id: "org.example.app".into(),
        title: "App".into(),
        identifier: "app-001".into(),
    }];
    let compositor = MockExtCompositor::new(128, 96).with_toplevels(toplevels);
    with_mock_wayland(compositor, || {
        let result = ExtImageCaptureBackend::new(CaptureSource::NewToplevel {
            known_identifiers: Vec::new(),
        });
        assert!(result.is_ok(), "expected Ok, got: {:?}", result.err());
    });
}

// ---------------------------------------------------------------------------
// Test: NewToplevel where all are known (no new found)
// ---------------------------------------------------------------------------

#[test]
fn ext_backend_new_toplevel_all_known() {
    let toplevels = vec![MockToplevel {
        app_id: "org.example.app".into(),
        title: "App".into(),
        identifier: "app-001".into(),
    }];
    let compositor = MockExtCompositor::new(128, 96).with_toplevels(toplevels);
    with_mock_wayland(compositor, || {
        let result = ExtImageCaptureBackend::new(CaptureSource::NewToplevel {
            known_identifiers: vec!["app-001".to_string()],
        });
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("no new toplevel"),
            "expected 'no new toplevel' in error: {msg}"
        );
    });
}

// ---------------------------------------------------------------------------
// Test: is_available()
// ---------------------------------------------------------------------------

#[test]
fn ext_backend_is_available() {
    let compositor = MockExtCompositor::new(128, 96);
    with_mock_wayland(compositor, || {
        let conn = wayland_client::Connection::connect_to_env().unwrap();
        assert!(ExtImageCaptureBackend::is_available(&conn));
    });
}

// ---------------------------------------------------------------------------
// Test: enumerate_toplevels()
// ---------------------------------------------------------------------------

#[test]
fn ext_enumerate_toplevels() {
    let toplevels = vec![
        MockToplevel {
            app_id: "org.mozilla.firefox".into(),
            title: "Firefox".into(),
            identifier: "ff-001".into(),
        },
        MockToplevel {
            app_id: "org.gnome.Terminal".into(),
            title: "Terminal".into(),
            identifier: "term-002".into(),
        },
    ];
    let compositor = MockExtCompositor::new(128, 96).with_toplevels(toplevels);
    with_mock_wayland(compositor, || {
        let result = enumerate_toplevels();
        assert!(result.is_ok(), "expected Ok, got: {:?}", result.err());
        let list = result.unwrap();
        assert_eq!(list.len(), 2);

        // Check first toplevel.
        let ff = list.iter().find(|t| t.app_id == "org.mozilla.firefox");
        assert!(ff.is_some(), "firefox not found in list");
        let ff = ff.unwrap();
        assert_eq!(ff.title, "Firefox");
        assert_eq!(ff.identifier, "ff-001");

        // Check second toplevel.
        let term = list.iter().find(|t| t.app_id == "org.gnome.Terminal");
        assert!(term.is_some(), "terminal not found in list");
        let term = term.unwrap();
        assert_eq!(term.title, "Terminal");
        assert_eq!(term.identifier, "term-002");
    });
}

// ---------------------------------------------------------------------------
// Test: enumerate_toplevels() with no toplevels returns empty vec
// ---------------------------------------------------------------------------

#[test]
fn ext_enumerate_toplevels_empty() {
    let compositor = MockExtCompositor::new(128, 96);
    with_mock_wayland(compositor, || {
        let result = enumerate_toplevels();
        assert!(result.is_ok());
        let list = result.unwrap();
        assert!(list.is_empty());
    });
}

// ---------------------------------------------------------------------------
// Test: Capture a frame via next_frame()
// ---------------------------------------------------------------------------

#[test]
fn ext_backend_capture_frame() {
    let compositor = MockExtCompositor::new(64, 48);
    with_mock_wayland(compositor, || {
        let mut backend = ExtImageCaptureBackend::new(CaptureSource::Output(None)).unwrap();
        let frame = backend.next_frame().unwrap();
        assert_eq!(frame.width, 64);
        assert_eq!(frame.height, 48);
        assert_eq!(frame.stride, 64 * 4);
        assert_eq!(frame.format, PixelFormat::Xrgb8888);
        assert!(!frame.damage.is_empty());
        assert_eq!(frame.data.len(), (64 * 4 * 48) as usize);
        // Presentation time: 1 * 1_000_000_000 + 500_000
        assert_eq!(frame.timestamp_ns, 1_000_000_000 + 500_000);
    });
}

// ---------------------------------------------------------------------------
// Test: Multiple sequential captures
// ---------------------------------------------------------------------------

#[test]
fn ext_backend_multiple_captures() {
    let compositor = MockExtCompositor::new(32, 24);
    with_mock_wayland(compositor, || {
        let mut backend = ExtImageCaptureBackend::new(CaptureSource::Output(None)).unwrap();
        for _ in 0..3 {
            let frame = backend.next_frame().unwrap();
            assert_eq!(frame.width, 32);
            assert_eq!(frame.height, 24);
        }
    });
}

// ---------------------------------------------------------------------------
// Test: stop() makes next_frame return SessionEnded
// ---------------------------------------------------------------------------

#[test]
fn ext_backend_stop() {
    let compositor = MockExtCompositor::new(64, 48);
    with_mock_wayland(compositor, || {
        let mut backend = ExtImageCaptureBackend::new(CaptureSource::Output(None)).unwrap();
        // Capture one frame successfully.
        let _ = backend.next_frame().unwrap();
        // Stop the backend.
        backend.stop();
        // Next frame should return SessionEnded.
        let result = backend.next_frame();
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("session ended"),
            "expected 'session ended' in error: {msg}"
        );
    });
}

// ---------------------------------------------------------------------------
// Test: Backend name
// ---------------------------------------------------------------------------

#[test]
fn ext_backend_name() {
    let compositor = MockExtCompositor::new(64, 48);
    with_mock_wayland(compositor, || {
        let backend = ExtImageCaptureBackend::new(CaptureSource::Output(None)).unwrap();
        assert_eq!(backend.name(), "ext-image-capture");
    });
}

// ---------------------------------------------------------------------------
// Test: Toplevel with second toplevel that also has app_id matching
// ---------------------------------------------------------------------------

#[test]
fn ext_backend_toplevel_second_app() {
    let toplevels = vec![
        MockToplevel {
            app_id: "org.gnome.Terminal".into(),
            title: "Terminal".into(),
            identifier: "term-001".into(),
        },
        MockToplevel {
            app_id: "org.gnome.Nautilus".into(),
            title: "Files".into(),
            identifier: "naut-001".into(),
        },
    ];
    let compositor = MockExtCompositor::new(128, 96).with_toplevels(toplevels);
    with_mock_wayland(compositor, || {
        let result =
            ExtImageCaptureBackend::new(CaptureSource::Toplevel("org.gnome.Nautilus".to_string()));
        assert!(result.is_ok(), "expected Ok, got: {:?}", result.err());
    });
}

// ---------------------------------------------------------------------------
// Test: ToplevelInfo derives (Debug, Clone, PartialEq, Eq, Hash)
// ---------------------------------------------------------------------------

#[test]
fn toplevel_info_derives() {
    use std::collections::HashSet;

    let a = ToplevelInfo {
        app_id: "a".into(),
        title: "t".into(),
        identifier: "i".into(),
    };
    let b = a.clone();
    assert_eq!(a, b);

    let c = ToplevelInfo {
        app_id: "x".into(),
        title: "t".into(),
        identifier: "i".into(),
    };
    assert_ne!(a, c);

    let dbg = format!("{:?}", a);
    assert!(dbg.contains("app_id"));

    let mut set = HashSet::new();
    set.insert(a.clone());
    set.insert(b);
    assert_eq!(set.len(), 1);
    set.insert(c);
    assert_eq!(set.len(), 2);
}

// ---------------------------------------------------------------------------
// Test: Capture damage rect values
// ---------------------------------------------------------------------------

#[test]
fn ext_backend_damage_rects() {
    let compositor = MockExtCompositor::new(64, 48);
    with_mock_wayland(compositor, || {
        let mut backend = ExtImageCaptureBackend::new(CaptureSource::Output(None)).unwrap();
        let frame = backend.next_frame().unwrap();
        // Mock sends full-frame damage.
        assert_eq!(frame.damage.len(), 1);
        let d = &frame.damage[0];
        assert_eq!(d.x, 0);
        assert_eq!(d.y, 0);
        assert_eq!(d.width, 64);
        assert_eq!(d.height, 48);
    });
}

// ---------------------------------------------------------------------------
// Test: Capture with toplevel source then capture frame
// ---------------------------------------------------------------------------

#[test]
fn ext_backend_toplevel_capture_frame() {
    let toplevels = vec![MockToplevel {
        app_id: "org.example.app".into(),
        title: "Test App".into(),
        identifier: "test-001".into(),
    }];
    let compositor = MockExtCompositor::new(100, 75).with_toplevels(toplevels);
    with_mock_wayland(compositor, || {
        let mut backend =
            ExtImageCaptureBackend::new(CaptureSource::Toplevel("org.example.app".to_string()))
                .unwrap();
        let frame = backend.next_frame().unwrap();
        assert_eq!(frame.width, 100);
        assert_eq!(frame.height, 75);
    });
}

// ---------------------------------------------------------------------------
// Test: NewToplevel capture then frame
// ---------------------------------------------------------------------------

#[test]
fn ext_backend_new_toplevel_capture_frame() {
    let toplevels = vec![
        MockToplevel {
            app_id: "old.app".into(),
            title: "Old".into(),
            identifier: "old-001".into(),
        },
        MockToplevel {
            app_id: "new.app".into(),
            title: "New".into(),
            identifier: "new-001".into(),
        },
    ];
    let compositor = MockExtCompositor::new(80, 60).with_toplevels(toplevels);
    with_mock_wayland(compositor, || {
        let mut backend = ExtImageCaptureBackend::new(CaptureSource::NewToplevel {
            known_identifiers: vec!["old-001".to_string()],
        })
        .unwrap();
        let frame = backend.next_frame().unwrap();
        assert_eq!(frame.width, 80);
        assert_eq!(frame.height, 60);
    });
}

// ---------------------------------------------------------------------------
// Tests for the detect module via with_mock_wayland infrastructure.
// ---------------------------------------------------------------------------

use remoteway_capture::detect::{
    detect_backend, detect_new_toplevel_backend, detect_toplevel_backend, is_capture_available,
};

#[test]
fn detect_is_capture_available_with_mock() {
    let compositor = MockExtCompositor::new(64, 48);
    with_mock_wayland(compositor, || {
        assert!(is_capture_available());
    });
}

#[test]
fn detect_backend_succeeds_with_mock() {
    let compositor = MockExtCompositor::new(128, 96);
    with_mock_wayland(compositor, || {
        let result = detect_backend(None);
        assert!(
            result.is_ok(),
            "detect_backend failed: {:?}",
            result.err().map(|e| e.to_string())
        );
        let backend = result.unwrap();
        // The backend may be either ext-image-capture or wlr-screencopy depending
        // on what the mock advertises. Just check it has a name.
        assert!(!backend.name().is_empty());
    });
}

#[test]
fn detect_toplevel_backend_with_mock() {
    let toplevels = vec![MockToplevel {
        app_id: "com.example.app".into(),
        title: "App".into(),
        identifier: "app-001".into(),
    }];
    let compositor = MockExtCompositor::new(128, 96).with_toplevels(toplevels);
    with_mock_wayland(compositor, || {
        let result = detect_toplevel_backend("com.example.app");
        assert!(
            result.is_ok(),
            "detect_toplevel_backend failed: {:?}",
            result.err().map(|e| e.to_string())
        );
    });
}

#[test]
fn detect_toplevel_backend_unknown_app_id() {
    let compositor = MockExtCompositor::new(64, 48);
    with_mock_wayland(compositor, || {
        let result = detect_toplevel_backend("does.not.exist");
        assert!(result.is_err());
    });
}

#[test]
fn detect_new_toplevel_backend_with_mock() {
    let toplevels = vec![
        MockToplevel {
            app_id: "old.app".into(),
            title: "Old".into(),
            identifier: "old-001".into(),
        },
        MockToplevel {
            app_id: "new.app".into(),
            title: "New".into(),
            identifier: "new-001".into(),
        },
    ];
    let compositor = MockExtCompositor::new(128, 96).with_toplevels(toplevels);
    with_mock_wayland(compositor, || {
        let result = detect_new_toplevel_backend(&["old-001".to_string()]);
        assert!(result.is_ok());
    });
}

#[test]
fn detect_new_toplevel_backend_no_new_toplevels() {
    let toplevels = vec![MockToplevel {
        app_id: "only.app".into(),
        title: "Only".into(),
        identifier: "only-001".into(),
    }];
    let compositor = MockExtCompositor::new(128, 96).with_toplevels(toplevels);
    with_mock_wayland(compositor, || {
        let result = detect_new_toplevel_backend(&["only-001".to_string()]);
        assert!(result.is_err());
    });
}

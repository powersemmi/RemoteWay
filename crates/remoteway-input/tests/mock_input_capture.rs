//! Integration tests for `InputCapture` using a mock Wayland compositor that
//! exposes `wl_seat`, `wl_pointer`, and `wl_keyboard` over a real listening
//! socket. Sets `WAYLAND_DISPLAY` so the actual `InputCapture::new()` code
//! path is exercised.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use wayland_server::protocol::{wl_keyboard, wl_pointer, wl_seat};
use wayland_server::{
    Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, ListeningSocket, New,
};

/// Mutex to serialize tests that modify the WAYLAND_DISPLAY env var.
static WAYLAND_DISPLAY_LOCK: Mutex<()> = Mutex::new(());

struct MockCaptureCompositor {
    pointers: Vec<wl_pointer::WlPointer>,
    keyboards: Vec<wl_keyboard::WlKeyboard>,
}

impl MockCaptureCompositor {
    fn new() -> Self {
        Self {
            pointers: Vec::new(),
            keyboards: Vec::new(),
        }
    }
}

impl GlobalDispatch<wl_seat::WlSeat, ()> for MockCaptureCompositor {
    fn bind(
        _state: &mut Self,
        _handle: &DisplayHandle,
        _client: &Client,
        resource: New<wl_seat::WlSeat>,
        _data: &(),
        data_init: &mut DataInit<'_, Self>,
    ) {
        let seat = data_init.init(resource, ());
        seat.capabilities(wl_seat::Capability::Pointer | wl_seat::Capability::Keyboard);
        seat.name("seat0".to_string());
    }
}

impl Dispatch<wl_seat::WlSeat, ()> for MockCaptureCompositor {
    fn request(
        state: &mut Self,
        _client: &Client,
        _resource: &wl_seat::WlSeat,
        request: wl_seat::Request,
        _data: &(),
        _handle: &DisplayHandle,
        data_init: &mut DataInit<'_, Self>,
    ) {
        match request {
            wl_seat::Request::GetPointer { id } => {
                let pointer = data_init.init(id, ());
                state.pointers.push(pointer);
            }
            wl_seat::Request::GetKeyboard { id } => {
                let keyboard = data_init.init(id, ());
                state.keyboards.push(keyboard);
            }
            _ => {}
        }
    }
}

impl Dispatch<wl_pointer::WlPointer, ()> for MockCaptureCompositor {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &wl_pointer::WlPointer,
        _request: wl_pointer::Request,
        _data: &(),
        _handle: &DisplayHandle,
        _data_init: &mut DataInit<'_, Self>,
    ) {
    }
}

impl Dispatch<wl_keyboard::WlKeyboard, ()> for MockCaptureCompositor {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &wl_keyboard::WlKeyboard,
        _request: wl_keyboard::Request,
        _data: &(),
        _handle: &DisplayHandle,
        _data_init: &mut DataInit<'_, Self>,
    ) {
    }
}

struct CaptureMock {
    stop: Arc<AtomicBool>,
    server_thread: Option<std::thread::JoinHandle<()>>,
    _guard: std::sync::MutexGuard<'static, ()>,
    old_display: Option<String>,
}

impl CaptureMock {
    fn new() -> Self {
        let guard = WAYLAND_DISPLAY_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let display = wayland_server::Display::<MockCaptureCompositor>::new().unwrap();
        let dh = display.handle();
        dh.create_global::<MockCaptureCompositor, wl_seat::WlSeat, ()>(8, ());

        let id = std::process::id();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos();
        let socket_name = format!("remoteway-capture-test-{}-{}", id, ts);
        let listener = ListeningSocket::bind(&socket_name).unwrap();

        let old_display = std::env::var("WAYLAND_DISPLAY").ok();
        unsafe { std::env::set_var("WAYLAND_DISPLAY", &socket_name) };

        let stop = Arc::new(AtomicBool::new(false));
        let stop_server = Arc::clone(&stop);

        let server_thread = std::thread::spawn(move || {
            let mut display = display;
            let mut compositor = MockCaptureCompositor::new();
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

impl Drop for CaptureMock {
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
fn input_capture_new_succeeds_with_mock() {
    let _mock = CaptureMock::new();
    let result = remoteway_input::capture::InputCapture::new();
    assert!(
        result.is_ok(),
        "InputCapture::new failed: {:?}",
        result.err().map(|e| e.to_string())
    );
}

#[test]
fn input_capture_poll_events_returns_empty_initially() {
    let _mock = CaptureMock::new();
    let mut cap = remoteway_input::capture::InputCapture::new().unwrap();
    let events = cap.poll_events().unwrap();
    // No server-side events queued yet.
    assert!(events.is_empty());
}

#[test]
fn input_capture_poll_events_repeated_calls() {
    let _mock = CaptureMock::new();
    let mut cap = remoteway_input::capture::InputCapture::new().unwrap();
    // Multiple polls should each succeed without error and return empty.
    for _ in 0..5 {
        let events = cap.poll_events().unwrap();
        assert!(events.is_empty());
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}

#[test]
fn input_capture_drop_is_clean() {
    let _mock = CaptureMock::new();
    let cap = remoteway_input::capture::InputCapture::new().unwrap();
    drop(cap);
    // No panic on drop.
}

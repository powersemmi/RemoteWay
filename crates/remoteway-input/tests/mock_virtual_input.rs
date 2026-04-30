//! Integration test using an in-process wayland-server mock that implements
//! wl_seat, zwlr_virtual_pointer_manager_v1, and zwp_virtual_keyboard_manager_v1
//! for full virtual input protocol round-trip testing.

use std::os::unix::net::UnixStream;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use wayland_protocols_misc::zwp_virtual_keyboard_v1::server::{
    zwp_virtual_keyboard_manager_v1, zwp_virtual_keyboard_v1,
};
use wayland_protocols_wlr::virtual_pointer::v1::server::{
    zwlr_virtual_pointer_manager_v1, zwlr_virtual_pointer_v1,
};
use wayland_server::protocol::wl_seat;
use wayland_server::{Client, Dispatch, DisplayHandle, GlobalDispatch, New};

// --- Mock compositor state ---

struct MockInputCompositor {
    /// Counter for virtual pointer motion_absolute requests received.
    motion_count: Arc<AtomicU32>,
    /// Counter for virtual pointer button requests received.
    button_count: Arc<AtomicU32>,
    /// Counter for virtual pointer axis requests received.
    axis_count: Arc<AtomicU32>,
    /// Counter for virtual keyboard key requests received.
    key_count: Arc<AtomicU32>,
    /// Counter for virtual keyboard keymap requests received.
    keymap_count: Arc<AtomicU32>,
}

type MockCounters = (
    Arc<AtomicU32>,
    Arc<AtomicU32>,
    Arc<AtomicU32>,
    Arc<AtomicU32>,
    Arc<AtomicU32>,
);

impl MockInputCompositor {
    fn new() -> (Self, MockCounters) {
        let motion = Arc::new(AtomicU32::new(0));
        let button = Arc::new(AtomicU32::new(0));
        let axis = Arc::new(AtomicU32::new(0));
        let key = Arc::new(AtomicU32::new(0));
        let keymap = Arc::new(AtomicU32::new(0));
        (
            Self {
                motion_count: Arc::clone(&motion),
                button_count: Arc::clone(&button),
                axis_count: Arc::clone(&axis),
                key_count: Arc::clone(&key),
                keymap_count: Arc::clone(&keymap),
            },
            (motion, button, axis, key, keymap),
        )
    }
}

// --- GlobalDispatch: wl_seat ---

impl GlobalDispatch<wl_seat::WlSeat, ()> for MockInputCompositor {
    fn bind(
        _state: &mut Self,
        _handle: &DisplayHandle,
        _client: &Client,
        resource: New<wl_seat::WlSeat>,
        _data: &(),
        data_init: &mut wayland_server::DataInit<'_, Self>,
    ) {
        let seat = data_init.init(resource, ());
        seat.capabilities(wl_seat::Capability::Pointer | wl_seat::Capability::Keyboard);
    }
}

impl Dispatch<wl_seat::WlSeat, ()> for MockInputCompositor {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &wl_seat::WlSeat,
        _request: wl_seat::Request,
        _data: &(),
        _handle: &DisplayHandle,
        _data_init: &mut wayland_server::DataInit<'_, Self>,
    ) {
    }
}

// --- GlobalDispatch: zwlr_virtual_pointer_manager_v1 ---

impl GlobalDispatch<zwlr_virtual_pointer_manager_v1::ZwlrVirtualPointerManagerV1, ()>
    for MockInputCompositor
{
    fn bind(
        _state: &mut Self,
        _handle: &DisplayHandle,
        _client: &Client,
        resource: New<zwlr_virtual_pointer_manager_v1::ZwlrVirtualPointerManagerV1>,
        _data: &(),
        data_init: &mut wayland_server::DataInit<'_, Self>,
    ) {
        data_init.init(resource, ());
    }
}

impl Dispatch<zwlr_virtual_pointer_manager_v1::ZwlrVirtualPointerManagerV1, ()>
    for MockInputCompositor
{
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &zwlr_virtual_pointer_manager_v1::ZwlrVirtualPointerManagerV1,
        request: zwlr_virtual_pointer_manager_v1::Request,
        _data: &(),
        _handle: &DisplayHandle,
        data_init: &mut wayland_server::DataInit<'_, Self>,
    ) {
        match request {
            zwlr_virtual_pointer_manager_v1::Request::CreateVirtualPointer { id, .. } => {
                data_init.init(id, ());
            }
            zwlr_virtual_pointer_manager_v1::Request::CreateVirtualPointerWithOutput {
                id, ..
            } => {
                data_init.init(id, ());
            }
            _ => {}
        }
    }
}

impl Dispatch<zwlr_virtual_pointer_v1::ZwlrVirtualPointerV1, ()> for MockInputCompositor {
    fn request(
        state: &mut Self,
        _client: &Client,
        _resource: &zwlr_virtual_pointer_v1::ZwlrVirtualPointerV1,
        request: zwlr_virtual_pointer_v1::Request,
        _data: &(),
        _handle: &DisplayHandle,
        _data_init: &mut wayland_server::DataInit<'_, Self>,
    ) {
        match request {
            zwlr_virtual_pointer_v1::Request::MotionAbsolute { .. } => {
                state.motion_count.fetch_add(1, Ordering::Relaxed);
            }
            zwlr_virtual_pointer_v1::Request::Button { .. } => {
                state.button_count.fetch_add(1, Ordering::Relaxed);
            }
            zwlr_virtual_pointer_v1::Request::Axis { .. } => {
                state.axis_count.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
    }
}

// --- GlobalDispatch: zwp_virtual_keyboard_manager_v1 ---

impl GlobalDispatch<zwp_virtual_keyboard_manager_v1::ZwpVirtualKeyboardManagerV1, ()>
    for MockInputCompositor
{
    fn bind(
        _state: &mut Self,
        _handle: &DisplayHandle,
        _client: &Client,
        resource: New<zwp_virtual_keyboard_manager_v1::ZwpVirtualKeyboardManagerV1>,
        _data: &(),
        data_init: &mut wayland_server::DataInit<'_, Self>,
    ) {
        data_init.init(resource, ());
    }
}

impl Dispatch<zwp_virtual_keyboard_manager_v1::ZwpVirtualKeyboardManagerV1, ()>
    for MockInputCompositor
{
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &zwp_virtual_keyboard_manager_v1::ZwpVirtualKeyboardManagerV1,
        request: zwp_virtual_keyboard_manager_v1::Request,
        _data: &(),
        _handle: &DisplayHandle,
        data_init: &mut wayland_server::DataInit<'_, Self>,
    ) {
        if let zwp_virtual_keyboard_manager_v1::Request::CreateVirtualKeyboard { id, .. } = request
        {
            data_init.init(id, ());
        }
    }
}

impl Dispatch<zwp_virtual_keyboard_v1::ZwpVirtualKeyboardV1, ()> for MockInputCompositor {
    fn request(
        state: &mut Self,
        _client: &Client,
        _resource: &zwp_virtual_keyboard_v1::ZwpVirtualKeyboardV1,
        request: zwp_virtual_keyboard_v1::Request,
        _data: &(),
        _handle: &DisplayHandle,
        _data_init: &mut wayland_server::DataInit<'_, Self>,
    ) {
        match request {
            zwp_virtual_keyboard_v1::Request::Key { .. } => {
                state.key_count.fetch_add(1, Ordering::Relaxed);
            }
            zwp_virtual_keyboard_v1::Request::Keymap { .. } => {
                state.keymap_count.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
    }
}

// --- Helper: create mock compositor and client connection ---

struct MockSetup {
    motion_count: Arc<AtomicU32>,
    button_count: Arc<AtomicU32>,
    axis_count: Arc<AtomicU32>,
    key_count: Arc<AtomicU32>,
    keymap_count: Arc<AtomicU32>,
    stop: Arc<AtomicBool>,
    server_thread: Option<std::thread::JoinHandle<()>>,
    client_conn: wayland_client::Connection,
}

impl MockSetup {
    fn new() -> Self {
        let display = wayland_server::Display::<MockInputCompositor>::new().unwrap();
        let mut dh = display.handle();

        dh.create_global::<MockInputCompositor, wl_seat::WlSeat, ()>(8, ());
        dh.create_global::<MockInputCompositor, zwlr_virtual_pointer_manager_v1::ZwlrVirtualPointerManagerV1, ()>(2, ());
        dh.create_global::<MockInputCompositor, zwp_virtual_keyboard_manager_v1::ZwpVirtualKeyboardManagerV1, ()>(1, ());

        let (client_stream, server_stream) = UnixStream::pair().unwrap();
        dh.insert_client(server_stream, Arc::new(())).unwrap();
        let client_conn = wayland_client::Connection::from_socket(client_stream).unwrap();

        let (mut compositor, (motion, button, axis, key, keymap)) = MockInputCompositor::new();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_server = Arc::clone(&stop);

        let server_thread = std::thread::spawn(move || {
            let mut display = display;
            while !stop_server.load(Ordering::Relaxed) {
                display.dispatch_clients(&mut compositor).unwrap();
                display.flush_clients().unwrap();
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        });

        Self {
            motion_count: motion,
            button_count: button,
            axis_count: axis,
            key_count: key,
            keymap_count: keymap,
            stop,
            server_thread: Some(server_thread),
            client_conn,
        }
    }

    fn shutdown(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.server_thread.take() {
            let _ = h.join();
        }
    }
}

impl Drop for MockSetup {
    fn drop(&mut self) {
        self.shutdown();
    }
}

// --- Client-side Dispatch implementations for VirtualInput ---

// VirtualInput uses its own InjectState which implements all needed Dispatch traits,
// but we need to connect via the mock socket. We test by creating VirtualInput
// using the mock connection directly via a helper that sets WAYLAND_DISPLAY.

// Since VirtualInput::new() uses Connection::connect_to_env(), we cannot easily
// pass a custom connection. Instead, we test the protocol at a lower level.

// --- Tests ---

#[test]
fn mock_compositor_accepts_virtual_pointer_and_keyboard() {
    use wayland_client::protocol::wl_registry;
    use wayland_client::{Connection as CC, Dispatch as CD, QueueHandle};
    use wayland_protocols_misc::zwp_virtual_keyboard_v1::client::{
        zwp_virtual_keyboard_manager_v1 as vkm_c, zwp_virtual_keyboard_v1 as vk_c,
    };
    use wayland_protocols_wlr::virtual_pointer::v1::client::{
        zwlr_virtual_pointer_manager_v1 as vpm_c, zwlr_virtual_pointer_v1 as vp_c,
    };

    let mut mock = MockSetup::new();

    // Client state matching what VirtualInput uses.
    struct CS {
        seat: Option<wayland_client::protocol::wl_seat::WlSeat>,
        vp_mgr: Option<vpm_c::ZwlrVirtualPointerManagerV1>,
        vk_mgr: Option<vkm_c::ZwpVirtualKeyboardManagerV1>,
        vp: Option<vp_c::ZwlrVirtualPointerV1>,
        vk: Option<vk_c::ZwpVirtualKeyboardV1>,
    }

    impl CD<wl_registry::WlRegistry, ()> for CS {
        fn event(
            state: &mut Self,
            registry: &wl_registry::WlRegistry,
            event: wl_registry::Event,
            _: &(),
            _: &CC,
            qh: &QueueHandle<Self>,
        ) {
            if let wl_registry::Event::Global {
                name,
                interface,
                version,
            } = event
            {
                match interface.as_str() {
                    "wl_seat" => {
                        state.seat = Some(registry.bind(name, version.min(1), qh, ()));
                    }
                    "zwlr_virtual_pointer_manager_v1" => {
                        state.vp_mgr = Some(registry.bind(name, version.min(2), qh, ()));
                    }
                    "zwp_virtual_keyboard_manager_v1" => {
                        state.vk_mgr = Some(registry.bind(name, version.min(1), qh, ()));
                    }
                    _ => {}
                }
            }
        }
    }

    impl CD<wayland_client::protocol::wl_seat::WlSeat, ()> for CS {
        fn event(
            _: &mut Self,
            _: &wayland_client::protocol::wl_seat::WlSeat,
            _: wayland_client::protocol::wl_seat::Event,
            _: &(),
            _: &CC,
            _: &QueueHandle<Self>,
        ) {
        }
    }
    impl CD<vpm_c::ZwlrVirtualPointerManagerV1, ()> for CS {
        fn event(
            _: &mut Self,
            _: &vpm_c::ZwlrVirtualPointerManagerV1,
            _: vpm_c::Event,
            _: &(),
            _: &CC,
            _: &QueueHandle<Self>,
        ) {
        }
    }
    impl CD<vp_c::ZwlrVirtualPointerV1, ()> for CS {
        fn event(
            _: &mut Self,
            _: &vp_c::ZwlrVirtualPointerV1,
            _: vp_c::Event,
            _: &(),
            _: &CC,
            _: &QueueHandle<Self>,
        ) {
        }
    }
    impl CD<vkm_c::ZwpVirtualKeyboardManagerV1, ()> for CS {
        fn event(
            _: &mut Self,
            _: &vkm_c::ZwpVirtualKeyboardManagerV1,
            _: vkm_c::Event,
            _: &(),
            _: &CC,
            _: &QueueHandle<Self>,
        ) {
        }
    }
    impl CD<vk_c::ZwpVirtualKeyboardV1, ()> for CS {
        fn event(
            _: &mut Self,
            _: &vk_c::ZwpVirtualKeyboardV1,
            _: vk_c::Event,
            _: &(),
            _: &CC,
            _: &QueueHandle<Self>,
        ) {
        }
    }

    let mut eq = mock.client_conn.new_event_queue::<CS>();
    let qh = eq.handle();
    let mut cs = CS {
        seat: None,
        vp_mgr: None,
        vk_mgr: None,
        vp: None,
        vk: None,
    };

    mock.client_conn.display().get_registry(&qh, ());
    eq.roundtrip(&mut cs).unwrap();

    assert!(cs.seat.is_some(), "wl_seat not bound");
    assert!(cs.vp_mgr.is_some(), "virtual pointer manager not bound");
    assert!(cs.vk_mgr.is_some(), "virtual keyboard manager not bound");

    // Create virtual devices.
    let seat = cs.seat.as_ref().unwrap();
    cs.vp = Some(
        cs.vp_mgr
            .as_ref()
            .unwrap()
            .create_virtual_pointer(Some(seat), &qh, ()),
    );
    cs.vk = Some(
        cs.vk_mgr
            .as_ref()
            .unwrap()
            .create_virtual_keyboard(seat, &qh, ()),
    );

    // Set keymap.
    let (keymap_fd, keymap_size) =
        remoteway_input::keymap::create_keymap_fd(remoteway_input::keymap::DEFAULT_KEYMAP).unwrap();
    use std::os::fd::AsFd;
    cs.vk
        .as_ref()
        .unwrap()
        .keymap(1, keymap_fd.as_fd(), keymap_size);

    eq.roundtrip(&mut cs).unwrap();
    // Server should have processed the dispatch.
    std::thread::sleep(std::time::Duration::from_millis(20));

    assert!(
        mock.keymap_count.load(Ordering::Relaxed) >= 1,
        "keymap was not sent"
    );

    // Send pointer motion.
    let vp = cs.vp.as_ref().unwrap();
    vp.motion_absolute(100, 500, 300, 0xFFFF, 0xFFFF);
    vp.frame();

    // Send pointer button.
    use wayland_client::protocol::wl_pointer;
    vp.button(101, 0x110, wl_pointer::ButtonState::Pressed);
    vp.frame();

    // Send pointer axis.
    vp.axis(102, wl_pointer::Axis::VerticalScroll, 5.0);
    vp.frame();

    // Send key.
    let vk = cs.vk.as_ref().unwrap();
    vk.key(103, 30, 1);

    // Flush and wait for server to process.
    let _ = mock.client_conn.flush();
    eq.roundtrip(&mut cs).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(30));

    assert!(
        mock.motion_count.load(Ordering::Relaxed) >= 1,
        "motion_absolute not received by mock"
    );
    assert!(
        mock.button_count.load(Ordering::Relaxed) >= 1,
        "button not received by mock"
    );
    assert!(
        mock.axis_count.load(Ordering::Relaxed) >= 1,
        "axis not received by mock"
    );
    assert!(
        mock.key_count.load(Ordering::Relaxed) >= 1,
        "key not received by mock"
    );

    mock.shutdown();
}

#[test]
fn mock_multiple_motion_events() {
    use wayland_client::protocol::wl_registry;
    use wayland_client::{Connection as CC, Dispatch as CD, QueueHandle};
    use wayland_protocols_wlr::virtual_pointer::v1::client::{
        zwlr_virtual_pointer_manager_v1 as vpm_c, zwlr_virtual_pointer_v1 as vp_c,
    };

    let mut mock = MockSetup::new();

    struct CS {
        seat: Option<wayland_client::protocol::wl_seat::WlSeat>,
        vp_mgr: Option<vpm_c::ZwlrVirtualPointerManagerV1>,
        vp: Option<vp_c::ZwlrVirtualPointerV1>,
    }

    impl CD<wl_registry::WlRegistry, ()> for CS {
        fn event(
            state: &mut Self,
            registry: &wl_registry::WlRegistry,
            event: wl_registry::Event,
            _: &(),
            _: &CC,
            qh: &QueueHandle<Self>,
        ) {
            if let wl_registry::Event::Global {
                name,
                interface,
                version,
            } = event
            {
                match interface.as_str() {
                    "wl_seat" => {
                        state.seat = Some(registry.bind(name, version.min(1), qh, ()));
                    }
                    "zwlr_virtual_pointer_manager_v1" => {
                        state.vp_mgr = Some(registry.bind(name, version.min(2), qh, ()));
                    }
                    _ => {}
                }
            }
        }
    }
    impl CD<wayland_client::protocol::wl_seat::WlSeat, ()> for CS {
        fn event(
            _: &mut Self,
            _: &wayland_client::protocol::wl_seat::WlSeat,
            _: wayland_client::protocol::wl_seat::Event,
            _: &(),
            _: &CC,
            _: &QueueHandle<Self>,
        ) {
        }
    }
    impl CD<vpm_c::ZwlrVirtualPointerManagerV1, ()> for CS {
        fn event(
            _: &mut Self,
            _: &vpm_c::ZwlrVirtualPointerManagerV1,
            _: vpm_c::Event,
            _: &(),
            _: &CC,
            _: &QueueHandle<Self>,
        ) {
        }
    }
    impl CD<vp_c::ZwlrVirtualPointerV1, ()> for CS {
        fn event(
            _: &mut Self,
            _: &vp_c::ZwlrVirtualPointerV1,
            _: vp_c::Event,
            _: &(),
            _: &CC,
            _: &QueueHandle<Self>,
        ) {
        }
    }

    let mut eq = mock.client_conn.new_event_queue::<CS>();
    let qh = eq.handle();
    let mut cs = CS {
        seat: None,
        vp_mgr: None,
        vp: None,
    };

    mock.client_conn.display().get_registry(&qh, ());
    eq.roundtrip(&mut cs).unwrap();

    let seat = cs.seat.as_ref().unwrap();
    cs.vp = Some(
        cs.vp_mgr
            .as_ref()
            .unwrap()
            .create_virtual_pointer(Some(seat), &qh, ()),
    );
    eq.roundtrip(&mut cs).unwrap();

    // Send 100 motion events.
    let vp = cs.vp.as_ref().unwrap();
    for i in 0..100 {
        vp.motion_absolute(i, i * 10, i * 20, 0xFFFF, 0xFFFF);
        vp.frame();
    }

    let _ = mock.client_conn.flush();
    eq.roundtrip(&mut cs).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(30));

    let count = mock.motion_count.load(Ordering::Relaxed);
    assert_eq!(count, 100, "expected 100 motions, got {count}");

    mock.shutdown();
}

#[test]
fn mock_multiple_button_events() {
    use wayland_client::protocol::{wl_pointer, wl_registry};
    use wayland_client::{Connection as CC, Dispatch as CD, QueueHandle};
    use wayland_protocols_wlr::virtual_pointer::v1::client::{
        zwlr_virtual_pointer_manager_v1 as vpm_c, zwlr_virtual_pointer_v1 as vp_c,
    };

    let mut mock = MockSetup::new();

    struct CS {
        seat: Option<wayland_client::protocol::wl_seat::WlSeat>,
        vp_mgr: Option<vpm_c::ZwlrVirtualPointerManagerV1>,
        vp: Option<vp_c::ZwlrVirtualPointerV1>,
    }

    impl CD<wl_registry::WlRegistry, ()> for CS {
        fn event(
            state: &mut Self,
            registry: &wl_registry::WlRegistry,
            event: wl_registry::Event,
            _: &(),
            _: &CC,
            qh: &QueueHandle<Self>,
        ) {
            if let wl_registry::Event::Global {
                name,
                interface,
                version,
            } = event
            {
                match interface.as_str() {
                    "wl_seat" => {
                        state.seat = Some(registry.bind(name, version.min(1), qh, ()));
                    }
                    "zwlr_virtual_pointer_manager_v1" => {
                        state.vp_mgr = Some(registry.bind(name, version.min(2), qh, ()));
                    }
                    _ => {}
                }
            }
        }
    }
    impl CD<wayland_client::protocol::wl_seat::WlSeat, ()> for CS {
        fn event(
            _: &mut Self,
            _: &wayland_client::protocol::wl_seat::WlSeat,
            _: wayland_client::protocol::wl_seat::Event,
            _: &(),
            _: &CC,
            _: &QueueHandle<Self>,
        ) {
        }
    }
    impl CD<vpm_c::ZwlrVirtualPointerManagerV1, ()> for CS {
        fn event(
            _: &mut Self,
            _: &vpm_c::ZwlrVirtualPointerManagerV1,
            _: vpm_c::Event,
            _: &(),
            _: &CC,
            _: &QueueHandle<Self>,
        ) {
        }
    }
    impl CD<vp_c::ZwlrVirtualPointerV1, ()> for CS {
        fn event(
            _: &mut Self,
            _: &vp_c::ZwlrVirtualPointerV1,
            _: vp_c::Event,
            _: &(),
            _: &CC,
            _: &QueueHandle<Self>,
        ) {
        }
    }

    let mut eq = mock.client_conn.new_event_queue::<CS>();
    let qh = eq.handle();
    let mut cs = CS {
        seat: None,
        vp_mgr: None,
        vp: None,
    };

    mock.client_conn.display().get_registry(&qh, ());
    eq.roundtrip(&mut cs).unwrap();

    let seat = cs.seat.as_ref().unwrap();
    cs.vp = Some(
        cs.vp_mgr
            .as_ref()
            .unwrap()
            .create_virtual_pointer(Some(seat), &qh, ()),
    );
    eq.roundtrip(&mut cs).unwrap();

    // Send multiple button presses and releases (simulating click sequences).
    let vp = cs.vp.as_ref().unwrap();
    for _ in 0..10 {
        vp.button(100, 0x110, wl_pointer::ButtonState::Pressed);
        vp.frame();
        vp.button(101, 0x110, wl_pointer::ButtonState::Released);
        vp.frame();
    }

    let _ = mock.client_conn.flush();
    eq.roundtrip(&mut cs).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(30));

    let count = mock.button_count.load(Ordering::Relaxed);
    assert_eq!(
        count, 20,
        "expected 20 buttons (10 press + 10 release), got {count}"
    );

    mock.shutdown();
}

#[test]
fn mock_multiple_axis_events() {
    use wayland_client::protocol::{wl_pointer, wl_registry};
    use wayland_client::{Connection as CC, Dispatch as CD, QueueHandle};
    use wayland_protocols_wlr::virtual_pointer::v1::client::{
        zwlr_virtual_pointer_manager_v1 as vpm_c, zwlr_virtual_pointer_v1 as vp_c,
    };

    let mut mock = MockSetup::new();

    struct CS {
        seat: Option<wayland_client::protocol::wl_seat::WlSeat>,
        vp_mgr: Option<vpm_c::ZwlrVirtualPointerManagerV1>,
        vp: Option<vp_c::ZwlrVirtualPointerV1>,
    }

    impl CD<wl_registry::WlRegistry, ()> for CS {
        fn event(
            state: &mut Self,
            registry: &wl_registry::WlRegistry,
            event: wl_registry::Event,
            _: &(),
            _: &CC,
            qh: &QueueHandle<Self>,
        ) {
            if let wl_registry::Event::Global {
                name,
                interface,
                version,
            } = event
            {
                match interface.as_str() {
                    "wl_seat" => {
                        state.seat = Some(registry.bind(name, version.min(1), qh, ()));
                    }
                    "zwlr_virtual_pointer_manager_v1" => {
                        state.vp_mgr = Some(registry.bind(name, version.min(2), qh, ()));
                    }
                    _ => {}
                }
            }
        }
    }
    impl CD<wayland_client::protocol::wl_seat::WlSeat, ()> for CS {
        fn event(
            _: &mut Self,
            _: &wayland_client::protocol::wl_seat::WlSeat,
            _: wayland_client::protocol::wl_seat::Event,
            _: &(),
            _: &CC,
            _: &QueueHandle<Self>,
        ) {
        }
    }
    impl CD<vpm_c::ZwlrVirtualPointerManagerV1, ()> for CS {
        fn event(
            _: &mut Self,
            _: &vpm_c::ZwlrVirtualPointerManagerV1,
            _: vpm_c::Event,
            _: &(),
            _: &CC,
            _: &QueueHandle<Self>,
        ) {
        }
    }
    impl CD<vp_c::ZwlrVirtualPointerV1, ()> for CS {
        fn event(
            _: &mut Self,
            _: &vp_c::ZwlrVirtualPointerV1,
            _: vp_c::Event,
            _: &(),
            _: &CC,
            _: &QueueHandle<Self>,
        ) {
        }
    }

    let mut eq = mock.client_conn.new_event_queue::<CS>();
    let qh = eq.handle();
    let mut cs = CS {
        seat: None,
        vp_mgr: None,
        vp: None,
    };

    mock.client_conn.display().get_registry(&qh, ());
    eq.roundtrip(&mut cs).unwrap();

    let seat = cs.seat.as_ref().unwrap();
    cs.vp = Some(
        cs.vp_mgr
            .as_ref()
            .unwrap()
            .create_virtual_pointer(Some(seat), &qh, ()),
    );
    eq.roundtrip(&mut cs).unwrap();

    let vp = cs.vp.as_ref().unwrap();
    // Send vertical and horizontal axis events.
    for i in 0..25 {
        let axis = if i % 2 == 0 {
            wl_pointer::Axis::VerticalScroll
        } else {
            wl_pointer::Axis::HorizontalScroll
        };
        vp.axis(i, axis, (i as f64) * 0.5);
        vp.frame();
    }

    let _ = mock.client_conn.flush();
    eq.roundtrip(&mut cs).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(30));

    let count = mock.axis_count.load(Ordering::Relaxed);
    assert_eq!(count, 25, "expected 25 axis events, got {count}");

    mock.shutdown();
}

#[test]
fn mock_multiple_key_events() {
    use wayland_client::protocol::wl_registry;
    use wayland_client::{Connection as CC, Dispatch as CD, QueueHandle};
    use wayland_protocols_misc::zwp_virtual_keyboard_v1::client::{
        zwp_virtual_keyboard_manager_v1 as vkm_c, zwp_virtual_keyboard_v1 as vk_c,
    };

    let mut mock = MockSetup::new();

    struct CS {
        seat: Option<wayland_client::protocol::wl_seat::WlSeat>,
        vk_mgr: Option<vkm_c::ZwpVirtualKeyboardManagerV1>,
        vk: Option<vk_c::ZwpVirtualKeyboardV1>,
    }

    impl CD<wl_registry::WlRegistry, ()> for CS {
        fn event(
            state: &mut Self,
            registry: &wl_registry::WlRegistry,
            event: wl_registry::Event,
            _: &(),
            _: &CC,
            qh: &QueueHandle<Self>,
        ) {
            if let wl_registry::Event::Global {
                name,
                interface,
                version,
            } = event
            {
                match interface.as_str() {
                    "wl_seat" => {
                        state.seat = Some(registry.bind(name, version.min(1), qh, ()));
                    }
                    "zwp_virtual_keyboard_manager_v1" => {
                        state.vk_mgr = Some(registry.bind(name, version.min(1), qh, ()));
                    }
                    _ => {}
                }
            }
        }
    }
    impl CD<wayland_client::protocol::wl_seat::WlSeat, ()> for CS {
        fn event(
            _: &mut Self,
            _: &wayland_client::protocol::wl_seat::WlSeat,
            _: wayland_client::protocol::wl_seat::Event,
            _: &(),
            _: &CC,
            _: &QueueHandle<Self>,
        ) {
        }
    }
    impl CD<vkm_c::ZwpVirtualKeyboardManagerV1, ()> for CS {
        fn event(
            _: &mut Self,
            _: &vkm_c::ZwpVirtualKeyboardManagerV1,
            _: vkm_c::Event,
            _: &(),
            _: &CC,
            _: &QueueHandle<Self>,
        ) {
        }
    }
    impl CD<vk_c::ZwpVirtualKeyboardV1, ()> for CS {
        fn event(
            _: &mut Self,
            _: &vk_c::ZwpVirtualKeyboardV1,
            _: vk_c::Event,
            _: &(),
            _: &CC,
            _: &QueueHandle<Self>,
        ) {
        }
    }

    let mut eq = mock.client_conn.new_event_queue::<CS>();
    let qh = eq.handle();
    let mut cs = CS {
        seat: None,
        vk_mgr: None,
        vk: None,
    };

    mock.client_conn.display().get_registry(&qh, ());
    eq.roundtrip(&mut cs).unwrap();

    let seat = cs.seat.as_ref().unwrap();
    cs.vk = Some(
        cs.vk_mgr
            .as_ref()
            .unwrap()
            .create_virtual_keyboard(seat, &qh, ()),
    );

    // Set keymap first (required before key events).
    let (keymap_fd, keymap_size) =
        remoteway_input::keymap::create_keymap_fd(remoteway_input::keymap::DEFAULT_KEYMAP).unwrap();
    use std::os::fd::AsFd;
    cs.vk
        .as_ref()
        .unwrap()
        .keymap(1, keymap_fd.as_fd(), keymap_size);

    eq.roundtrip(&mut cs).unwrap();

    // Send key press + release for multiple keys.
    let vk = cs.vk.as_ref().unwrap();
    let keys = [30, 31, 32, 33, 34]; // a, s, d, f, g
    for &key_code in &keys {
        vk.key(100, key_code, 1); // press
        vk.key(101, key_code, 0); // release
    }

    let _ = mock.client_conn.flush();
    eq.roundtrip(&mut cs).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(30));

    let count = mock.key_count.load(Ordering::Relaxed);
    assert_eq!(
        count, 10,
        "expected 10 key events (5 press + 5 release), got {count}"
    );

    mock.shutdown();
}

#[test]
fn mock_mixed_input_sequence() {
    use wayland_client::protocol::{wl_pointer, wl_registry};
    use wayland_client::{Connection as CC, Dispatch as CD, QueueHandle};
    use wayland_protocols_misc::zwp_virtual_keyboard_v1::client::{
        zwp_virtual_keyboard_manager_v1 as vkm_c, zwp_virtual_keyboard_v1 as vk_c,
    };
    use wayland_protocols_wlr::virtual_pointer::v1::client::{
        zwlr_virtual_pointer_manager_v1 as vpm_c, zwlr_virtual_pointer_v1 as vp_c,
    };

    let mut mock = MockSetup::new();

    struct CS {
        seat: Option<wayland_client::protocol::wl_seat::WlSeat>,
        vp_mgr: Option<vpm_c::ZwlrVirtualPointerManagerV1>,
        vk_mgr: Option<vkm_c::ZwpVirtualKeyboardManagerV1>,
        vp: Option<vp_c::ZwlrVirtualPointerV1>,
        vk: Option<vk_c::ZwpVirtualKeyboardV1>,
    }

    impl CD<wl_registry::WlRegistry, ()> for CS {
        fn event(
            state: &mut Self,
            registry: &wl_registry::WlRegistry,
            event: wl_registry::Event,
            _: &(),
            _: &CC,
            qh: &QueueHandle<Self>,
        ) {
            if let wl_registry::Event::Global {
                name,
                interface,
                version,
            } = event
            {
                match interface.as_str() {
                    "wl_seat" => {
                        state.seat = Some(registry.bind(name, version.min(1), qh, ()));
                    }
                    "zwlr_virtual_pointer_manager_v1" => {
                        state.vp_mgr = Some(registry.bind(name, version.min(2), qh, ()));
                    }
                    "zwp_virtual_keyboard_manager_v1" => {
                        state.vk_mgr = Some(registry.bind(name, version.min(1), qh, ()));
                    }
                    _ => {}
                }
            }
        }
    }

    impl CD<wayland_client::protocol::wl_seat::WlSeat, ()> for CS {
        fn event(
            _: &mut Self,
            _: &wayland_client::protocol::wl_seat::WlSeat,
            _: wayland_client::protocol::wl_seat::Event,
            _: &(),
            _: &CC,
            _: &QueueHandle<Self>,
        ) {
        }
    }
    impl CD<vpm_c::ZwlrVirtualPointerManagerV1, ()> for CS {
        fn event(
            _: &mut Self,
            _: &vpm_c::ZwlrVirtualPointerManagerV1,
            _: vpm_c::Event,
            _: &(),
            _: &CC,
            _: &QueueHandle<Self>,
        ) {
        }
    }
    impl CD<vp_c::ZwlrVirtualPointerV1, ()> for CS {
        fn event(
            _: &mut Self,
            _: &vp_c::ZwlrVirtualPointerV1,
            _: vp_c::Event,
            _: &(),
            _: &CC,
            _: &QueueHandle<Self>,
        ) {
        }
    }
    impl CD<vkm_c::ZwpVirtualKeyboardManagerV1, ()> for CS {
        fn event(
            _: &mut Self,
            _: &vkm_c::ZwpVirtualKeyboardManagerV1,
            _: vkm_c::Event,
            _: &(),
            _: &CC,
            _: &QueueHandle<Self>,
        ) {
        }
    }
    impl CD<vk_c::ZwpVirtualKeyboardV1, ()> for CS {
        fn event(
            _: &mut Self,
            _: &vk_c::ZwpVirtualKeyboardV1,
            _: vk_c::Event,
            _: &(),
            _: &CC,
            _: &QueueHandle<Self>,
        ) {
        }
    }

    let mut eq = mock.client_conn.new_event_queue::<CS>();
    let qh = eq.handle();
    let mut cs = CS {
        seat: None,
        vp_mgr: None,
        vk_mgr: None,
        vp: None,
        vk: None,
    };

    mock.client_conn.display().get_registry(&qh, ());
    eq.roundtrip(&mut cs).unwrap();

    let seat = cs.seat.as_ref().unwrap();
    cs.vp = Some(
        cs.vp_mgr
            .as_ref()
            .unwrap()
            .create_virtual_pointer(Some(seat), &qh, ()),
    );
    cs.vk = Some(
        cs.vk_mgr
            .as_ref()
            .unwrap()
            .create_virtual_keyboard(seat, &qh, ()),
    );

    // Set keymap.
    let (keymap_fd, keymap_size) =
        remoteway_input::keymap::create_keymap_fd(remoteway_input::keymap::DEFAULT_KEYMAP).unwrap();
    use std::os::fd::AsFd;
    cs.vk
        .as_ref()
        .unwrap()
        .keymap(1, keymap_fd.as_fd(), keymap_size);
    eq.roundtrip(&mut cs).unwrap();

    let vp = cs.vp.as_ref().unwrap();
    let vk = cs.vk.as_ref().unwrap();

    // Simulate a realistic input sequence: move cursor, click, type.
    // 1. Move cursor to target.
    vp.motion_absolute(1, 500, 300, 0xFFFF, 0xFFFF);
    vp.frame();
    vp.motion_absolute(2, 510, 305, 0xFFFF, 0xFFFF);
    vp.frame();

    // 2. Click (press + release).
    vp.button(3, 0x110, wl_pointer::ButtonState::Pressed);
    vp.frame();
    vp.button(4, 0x110, wl_pointer::ButtonState::Released);
    vp.frame();

    // 3. Scroll.
    vp.axis(5, wl_pointer::Axis::VerticalScroll, -3.0);
    vp.frame();

    // 4. Type "hello" (keycodes: h=35, e=26, l=38, l=38, o=32).
    for &key_code in &[35u32, 26, 38, 38, 32] {
        vk.key(10, key_code, 1); // press
        vk.key(11, key_code, 0); // release
    }

    let _ = mock.client_conn.flush();
    eq.roundtrip(&mut cs).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(30));

    assert!(mock.motion_count.load(Ordering::Relaxed) >= 2, "motions");
    assert!(mock.button_count.load(Ordering::Relaxed) >= 2, "buttons");
    assert!(mock.axis_count.load(Ordering::Relaxed) >= 1, "axis");
    assert!(mock.key_count.load(Ordering::Relaxed) >= 10, "keys");

    mock.shutdown();
}

#[test]
fn mock_shutdown_is_clean() {
    // Verify that MockSetup shutdown completes without error or hang.
    let mut mock = MockSetup::new();

    // Do nothing except create the setup and shut it down.
    mock.shutdown();

    // Calling shutdown again is safe.
    mock.shutdown();
}

#[test]
fn mock_compositor_drop_is_clean() {
    // Drop via the Drop impl (not explicit shutdown).
    let _mock = MockSetup::new();
    // Drop happens here automatically.
}

// ---------------------------------------------------------------------------
// Tests using WAYLAND_DISPLAY env var to exercise real VirtualInput::new()
// and dispatch_event() through the actual code paths.
// ---------------------------------------------------------------------------

use std::sync::Mutex;

/// Mutex to serialize tests that modify the WAYLAND_DISPLAY env var.
static WAYLAND_DISPLAY_LOCK: Mutex<()> = Mutex::new(());

struct ListeningMock {
    motion_count: Arc<AtomicU32>,
    button_count: Arc<AtomicU32>,
    axis_count: Arc<AtomicU32>,
    key_count: Arc<AtomicU32>,
    keymap_count: Arc<AtomicU32>,
    stop: Arc<AtomicBool>,
    server_thread: Option<std::thread::JoinHandle<()>>,
    _guard: std::sync::MutexGuard<'static, ()>,
    old_display: Option<String>,
}

impl ListeningMock {
    fn new() -> Self {
        use wayland_server::ListeningSocket;

        let guard = WAYLAND_DISPLAY_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let display = wayland_server::Display::<MockInputCompositor>::new().unwrap();
        let dh = display.handle();

        dh.create_global::<MockInputCompositor, wl_seat::WlSeat, ()>(8, ());
        dh.create_global::<MockInputCompositor, zwlr_virtual_pointer_manager_v1::ZwlrVirtualPointerManagerV1, ()>(2, ());
        dh.create_global::<MockInputCompositor, zwp_virtual_keyboard_manager_v1::ZwpVirtualKeyboardManagerV1, ()>(1, ());

        let id = std::process::id();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos();
        let socket_name = format!("remoteway-input-test-{}-{}", id, ts);
        let listener = ListeningSocket::bind(&socket_name).unwrap();

        let old_display = std::env::var("WAYLAND_DISPLAY").ok();
        unsafe { std::env::set_var("WAYLAND_DISPLAY", &socket_name) };

        let (mut compositor, (motion, button, axis, key, keymap)) = MockInputCompositor::new();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_server = Arc::clone(&stop);

        let server_thread = std::thread::spawn(move || {
            let mut display = display;
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
            motion_count: motion,
            button_count: button,
            axis_count: axis,
            key_count: key,
            keymap_count: keymap,
            stop,
            server_thread: Some(server_thread),
            _guard: guard,
            old_display,
        }
    }
}

impl Drop for ListeningMock {
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
fn virtual_input_new_succeeds_with_mock() {
    let _mock = ListeningMock::new();
    let result = remoteway_input::inject::VirtualInput::new();
    assert!(
        result.is_ok(),
        "VirtualInput::new() failed: {:?}",
        result.err().map(|e| e.to_string())
    );
}

#[test]
fn virtual_input_dispatch_pointer_motion() {
    use remoteway_proto::input::{InputEvent, PointerMotion};

    let mock = ListeningMock::new();
    let vi = remoteway_input::inject::VirtualInput::new().unwrap();

    let ev = InputEvent::pointer_motion(PointerMotion {
        surface_id: 0,
        _pad: 0,
        x: 1234.0,
        y: 5678.0,
    });
    vi.dispatch_event(&ev).unwrap();
    vi.flush();

    std::thread::sleep(std::time::Duration::from_millis(50));
    assert!(
        mock.motion_count.load(Ordering::Relaxed) >= 1,
        "motion event was not dispatched"
    );
}

#[test]
fn virtual_input_dispatch_pointer_button() {
    use remoteway_proto::input::{InputEvent, PointerButton};

    let mock = ListeningMock::new();
    let vi = remoteway_input::inject::VirtualInput::new().unwrap();

    // Pressed.
    let ev_press = InputEvent::pointer_button(PointerButton {
        button: 0x110,
        state: 1,
    });
    vi.dispatch_event(&ev_press).unwrap();

    // Released.
    let ev_release = InputEvent::pointer_button(PointerButton {
        button: 0x110,
        state: 0,
    });
    vi.dispatch_event(&ev_release).unwrap();
    vi.flush();

    std::thread::sleep(std::time::Duration::from_millis(50));
    assert!(mock.button_count.load(Ordering::Relaxed) >= 2);
}

#[test]
fn virtual_input_dispatch_pointer_axis_vertical() {
    use remoteway_proto::input::{InputEvent, PointerAxis};

    let mock = ListeningMock::new();
    let vi = remoteway_input::inject::VirtualInput::new().unwrap();

    let ev = InputEvent::pointer_axis(PointerAxis {
        axis: 0, // vertical
        _pad: [0; 3],
        value: -3.0,
    });
    vi.dispatch_event(&ev).unwrap();
    vi.flush();

    std::thread::sleep(std::time::Duration::from_millis(50));
    assert!(mock.axis_count.load(Ordering::Relaxed) >= 1);
}

#[test]
fn virtual_input_dispatch_pointer_axis_horizontal() {
    use remoteway_proto::input::{InputEvent, PointerAxis};

    let mock = ListeningMock::new();
    let vi = remoteway_input::inject::VirtualInput::new().unwrap();

    let ev = InputEvent::pointer_axis(PointerAxis {
        axis: 1, // horizontal
        _pad: [0; 3],
        value: 5.0,
    });
    vi.dispatch_event(&ev).unwrap();
    vi.flush();

    std::thread::sleep(std::time::Duration::from_millis(50));
    assert!(mock.axis_count.load(Ordering::Relaxed) >= 1);
}

#[test]
fn virtual_input_dispatch_key_event() {
    use remoteway_proto::input::{InputEvent, KeyEvent};

    let mock = ListeningMock::new();
    let vi = remoteway_input::inject::VirtualInput::new().unwrap();

    let ev = InputEvent::key(KeyEvent { key: 30, state: 1 });
    vi.dispatch_event(&ev).unwrap();
    vi.flush();

    std::thread::sleep(std::time::Duration::from_millis(50));
    assert!(mock.key_count.load(Ordering::Relaxed) >= 1);
    // Keymap should have been sent during VirtualInput::new().
    assert!(mock.keymap_count.load(Ordering::Relaxed) >= 1);
}

#[test]
fn virtual_input_dispatch_unknown_kind_returns_error() {
    use remoteway_proto::input::{InputEvent, KeyEvent};

    let _mock = ListeningMock::new();
    let vi = remoteway_input::inject::VirtualInput::new().unwrap();

    let mut ev = InputEvent::key(KeyEvent { key: 1, state: 0 });
    ev.kind = 199;
    let res = vi.dispatch_event(&ev);
    assert!(res.is_err());
}

#[test]
fn virtual_input_dispatch_mixed_events() {
    use remoteway_proto::input::{InputEvent, KeyEvent, PointerAxis, PointerButton, PointerMotion};

    let mock = ListeningMock::new();
    let vi = remoteway_input::inject::VirtualInput::new().unwrap();

    let events = [
        InputEvent::pointer_motion(PointerMotion {
            surface_id: 0,
            _pad: 0,
            x: 1.0,
            y: 2.0,
        }),
        InputEvent::pointer_button(PointerButton {
            button: 0x111,
            state: 1,
        }),
        InputEvent::pointer_axis(PointerAxis {
            axis: 0,
            _pad: [0; 3],
            value: 1.0,
        }),
        InputEvent::key(KeyEvent { key: 28, state: 1 }),
    ];

    for ev in &events {
        vi.dispatch_event(ev).unwrap();
    }
    vi.flush();

    std::thread::sleep(std::time::Duration::from_millis(80));
    assert!(mock.motion_count.load(Ordering::Relaxed) >= 1);
    assert!(mock.button_count.load(Ordering::Relaxed) >= 1);
    assert!(mock.axis_count.load(Ordering::Relaxed) >= 1);
    assert!(mock.key_count.load(Ordering::Relaxed) >= 1);
}

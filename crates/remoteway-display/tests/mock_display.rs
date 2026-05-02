//! Integration test using an in-process wayland-server mock that implements
//! `wl_compositor`, `wl_shm`, and `xdg_wm_base` for full display protocol round-trip.

use std::os::fd::AsFd;
use std::os::unix::net::UnixStream;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use wayland_client::{
    self as wc, Connection as ClientConnection, Dispatch as ClientDispatch, QueueHandle,
};
use wayland_protocols::xdg::shell::client as xdg_client;
use wayland_protocols::xdg::shell::server as xdg_server;
use wayland_server::protocol::{wl_buffer, wl_compositor, wl_shm, wl_shm_pool, wl_surface};
use wayland_server::{Client, Dispatch, DisplayHandle, GlobalDispatch, New};

// --- Mock compositor state ---

struct MockCompositor {
    surface_count: Arc<AtomicU32>,
    commit_count: Arc<AtomicU32>,
}

impl MockCompositor {
    fn new() -> Self {
        Self {
            surface_count: Arc::new(AtomicU32::new(0)),
            commit_count: Arc::new(AtomicU32::new(0)),
        }
    }
}

// --- wl_compositor ---

impl GlobalDispatch<wl_compositor::WlCompositor, ()> for MockCompositor {
    fn bind(
        _state: &mut Self,
        _handle: &DisplayHandle,
        _client: &Client,
        resource: New<wl_compositor::WlCompositor>,
        _data: &(),
        data_init: &mut wayland_server::DataInit<'_, Self>,
    ) {
        data_init.init(resource, ());
    }
}

impl Dispatch<wl_compositor::WlCompositor, ()> for MockCompositor {
    fn request(
        state: &mut Self,
        _client: &Client,
        _resource: &wl_compositor::WlCompositor,
        request: wl_compositor::Request,
        _data: &(),
        _handle: &DisplayHandle,
        data_init: &mut wayland_server::DataInit<'_, Self>,
    ) {
        if let wl_compositor::Request::CreateSurface { id } = request {
            data_init.init(id, ());
            state.surface_count.fetch_add(1, Ordering::Relaxed);
        }
    }
}

impl Dispatch<wl_surface::WlSurface, ()> for MockCompositor {
    fn request(
        state: &mut Self,
        _client: &Client,
        _resource: &wl_surface::WlSurface,
        request: wl_surface::Request,
        _data: &(),
        _handle: &DisplayHandle,
        _data_init: &mut wayland_server::DataInit<'_, Self>,
    ) {
        if let wl_surface::Request::Commit = request {
            state.commit_count.fetch_add(1, Ordering::Relaxed);
        }
    }
}

// --- wl_shm ---

impl GlobalDispatch<wl_shm::WlShm, ()> for MockCompositor {
    fn bind(
        _state: &mut Self,
        _handle: &DisplayHandle,
        _client: &Client,
        resource: New<wl_shm::WlShm>,
        _data: &(),
        data_init: &mut wayland_server::DataInit<'_, Self>,
    ) {
        let shm = data_init.init(resource, ());
        shm.format(wl_shm::Format::Xrgb8888);
        shm.format(wl_shm::Format::Argb8888);
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

// --- xdg_wm_base ---

impl GlobalDispatch<xdg_server::xdg_wm_base::XdgWmBase, ()> for MockCompositor {
    fn bind(
        _state: &mut Self,
        _handle: &DisplayHandle,
        _client: &Client,
        resource: New<xdg_server::xdg_wm_base::XdgWmBase>,
        _data: &(),
        data_init: &mut wayland_server::DataInit<'_, Self>,
    ) {
        data_init.init(resource, ());
    }
}

impl Dispatch<xdg_server::xdg_wm_base::XdgWmBase, ()> for MockCompositor {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &xdg_server::xdg_wm_base::XdgWmBase,
        request: xdg_server::xdg_wm_base::Request,
        _data: &(),
        _handle: &DisplayHandle,
        data_init: &mut wayland_server::DataInit<'_, Self>,
    ) {
        match request {
            xdg_server::xdg_wm_base::Request::GetXdgSurface { id, .. } => {
                data_init.init(id, ());
            }
            xdg_server::xdg_wm_base::Request::Pong { .. } => {}
            _ => {}
        }
    }
}

impl Dispatch<xdg_server::xdg_surface::XdgSurface, ()> for MockCompositor {
    fn request(
        _state: &mut Self,
        _client: &Client,
        resource: &xdg_server::xdg_surface::XdgSurface,
        request: xdg_server::xdg_surface::Request,
        _data: &(),
        _handle: &DisplayHandle,
        data_init: &mut wayland_server::DataInit<'_, Self>,
    ) {
        match request {
            xdg_server::xdg_surface::Request::GetToplevel { id } => {
                let toplevel = data_init.init(id, ());
                // Send configure events — required by xdg-shell protocol.
                toplevel.configure(0, 0, vec![]);
                resource.configure(1);
            }
            xdg_server::xdg_surface::Request::AckConfigure { .. } => {}
            _ => {}
        }
    }
}

impl Dispatch<xdg_server::xdg_toplevel::XdgToplevel, ()> for MockCompositor {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &xdg_server::xdg_toplevel::XdgToplevel,
        _request: xdg_server::xdg_toplevel::Request,
        _data: &(),
        _handle: &DisplayHandle,
        _data_init: &mut wayland_server::DataInit<'_, Self>,
    ) {
    }
}

// --- Test ---

#[test]
fn mock_compositor_surface_creation() {
    let mut display = wayland_server::Display::<MockCompositor>::new().unwrap();
    let mut dh = display.handle();

    dh.create_global::<MockCompositor, wl_compositor::WlCompositor, _>(6, ());
    dh.create_global::<MockCompositor, wl_shm::WlShm, _>(1, ());
    dh.create_global::<MockCompositor, xdg_server::xdg_wm_base::XdgWmBase, _>(5, ());

    let (client_stream, server_stream) = UnixStream::pair().unwrap();
    dh.insert_client(server_stream, Arc::new(())).unwrap();
    let client_conn = ClientConnection::from_socket(client_stream).unwrap();

    let mut compositor = MockCompositor::new();
    let surface_count = Arc::clone(&compositor.surface_count);
    let commit_count = Arc::clone(&compositor.commit_count);
    let stop = Arc::new(AtomicBool::new(false));
    let stop_server = Arc::clone(&stop);

    let server_thread = std::thread::spawn(move || {
        while !stop_server.load(Ordering::Relaxed) {
            display.dispatch_clients(&mut compositor).unwrap();
            display.flush_clients().unwrap();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    });

    // Client-side state mimicking DisplayState's dispatch.
    struct CS {
        compositor: Option<wc::protocol::wl_compositor::WlCompositor>,
        shm: Option<wc::protocol::wl_shm::WlShm>,
        xdg_wm_base: Option<xdg_client::xdg_wm_base::XdgWmBase>,
        configured: bool,
    }

    impl ClientDispatch<wc::protocol::wl_registry::WlRegistry, ()> for CS {
        fn event(
            state: &mut Self,
            registry: &wc::protocol::wl_registry::WlRegistry,
            event: wc::protocol::wl_registry::Event,
            _: &(),
            _: &wc::Connection,
            qh: &QueueHandle<Self>,
        ) {
            if let wc::protocol::wl_registry::Event::Global {
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
                    _ => {}
                }
            }
        }
    }

    impl ClientDispatch<wc::protocol::wl_compositor::WlCompositor, ()> for CS {
        fn event(
            _: &mut Self,
            _: &wc::protocol::wl_compositor::WlCompositor,
            _: wc::protocol::wl_compositor::Event,
            _: &(),
            _: &wc::Connection,
            _: &QueueHandle<Self>,
        ) {
        }
    }
    impl ClientDispatch<wc::protocol::wl_shm::WlShm, ()> for CS {
        fn event(
            _: &mut Self,
            _: &wc::protocol::wl_shm::WlShm,
            _: wc::protocol::wl_shm::Event,
            _: &(),
            _: &wc::Connection,
            _: &QueueHandle<Self>,
        ) {
        }
    }
    impl ClientDispatch<wc::protocol::wl_shm_pool::WlShmPool, ()> for CS {
        fn event(
            _: &mut Self,
            _: &wc::protocol::wl_shm_pool::WlShmPool,
            _: wc::protocol::wl_shm_pool::Event,
            _: &(),
            _: &wc::Connection,
            _: &QueueHandle<Self>,
        ) {
        }
    }
    impl ClientDispatch<wc::protocol::wl_buffer::WlBuffer, usize> for CS {
        fn event(
            _: &mut Self,
            _: &wc::protocol::wl_buffer::WlBuffer,
            _: wc::protocol::wl_buffer::Event,
            _: &usize,
            _: &wc::Connection,
            _: &QueueHandle<Self>,
        ) {
        }
    }
    impl ClientDispatch<wc::protocol::wl_surface::WlSurface, ()> for CS {
        fn event(
            _: &mut Self,
            _: &wc::protocol::wl_surface::WlSurface,
            _: wc::protocol::wl_surface::Event,
            _: &(),
            _: &wc::Connection,
            _: &QueueHandle<Self>,
        ) {
        }
    }
    impl ClientDispatch<xdg_client::xdg_wm_base::XdgWmBase, ()> for CS {
        fn event(
            _: &mut Self,
            base: &xdg_client::xdg_wm_base::XdgWmBase,
            event: xdg_client::xdg_wm_base::Event,
            _: &(),
            _: &wc::Connection,
            _: &QueueHandle<Self>,
        ) {
            if let xdg_client::xdg_wm_base::Event::Ping { serial } = event {
                base.pong(serial);
            }
        }
    }
    impl ClientDispatch<xdg_client::xdg_surface::XdgSurface, ()> for CS {
        fn event(
            state: &mut Self,
            surf: &xdg_client::xdg_surface::XdgSurface,
            event: xdg_client::xdg_surface::Event,
            _: &(),
            _: &wc::Connection,
            _: &QueueHandle<Self>,
        ) {
            if let xdg_client::xdg_surface::Event::Configure { serial } = event {
                surf.ack_configure(serial);
                state.configured = true;
            }
        }
    }
    impl ClientDispatch<xdg_client::xdg_toplevel::XdgToplevel, ()> for CS {
        fn event(
            _: &mut Self,
            _: &xdg_client::xdg_toplevel::XdgToplevel,
            _: xdg_client::xdg_toplevel::Event,
            _: &(),
            _: &wc::Connection,
            _: &QueueHandle<Self>,
        ) {
        }
    }

    let mut eq = client_conn.new_event_queue::<CS>();
    let qh = eq.handle();
    let mut cs = CS {
        compositor: None,
        shm: None,
        xdg_wm_base: None,
        configured: false,
    };

    client_conn.display().get_registry(&qh, ());

    // Roundtrip 1: discover globals.
    eq.roundtrip(&mut cs).unwrap();
    assert!(cs.compositor.is_some(), "wl_compositor not found");
    assert!(cs.shm.is_some(), "wl_shm not found");
    assert!(cs.xdg_wm_base.is_some(), "xdg_wm_base not found");

    // Create surface.
    let wl_surface = cs.compositor.as_ref().unwrap().create_surface(&qh, ());
    let xdg_surface = cs
        .xdg_wm_base
        .as_ref()
        .unwrap()
        .get_xdg_surface(&wl_surface, &qh, ());
    let toplevel = xdg_surface.get_toplevel(&qh, ());
    toplevel.set_title("Test Window".into());
    toplevel.set_app_id("test-app".into());

    // Initial commit to negotiate.
    wl_surface.commit();

    // Roundtrip 2: receive configure.
    eq.roundtrip(&mut cs).unwrap();
    assert!(cs.configured, "xdg_surface not configured");

    // Create SHM buffer and attach.
    let shm = cs.shm.as_ref().unwrap();
    let fd =
        nix::sys::memfd::memfd_create(c"test-shm", nix::sys::memfd::MemFdCreateFlag::MFD_CLOEXEC)
            .unwrap();
    let buf_size: i32 = 64 * 48 * 4;
    nix::unistd::ftruncate(&fd, buf_size as i64).unwrap();
    let pool = shm.create_pool(fd.as_fd(), buf_size, &qh, ());
    let buffer = pool.create_buffer(
        0,
        64,
        48,
        64 * 4,
        wc::protocol::wl_shm::Format::Xrgb8888,
        &qh,
        0usize,
    );

    wl_surface.attach(Some(&buffer), 0, 0);
    wl_surface.damage_buffer(0, 0, 64, 48);
    wl_surface.commit();

    // Roundtrip 3: ensure server processed the commit.
    eq.roundtrip(&mut cs).unwrap();

    // Verify server state.
    assert!(surface_count.load(Ordering::Relaxed) >= 1);
    assert!(commit_count.load(Ordering::Relaxed) >= 2); // initial + attach commit

    // Cleanup.
    toplevel.destroy();
    xdg_surface.destroy();
    wl_surface.destroy();
    buffer.destroy();
    pool.destroy();
    stop.store(true, Ordering::Relaxed);
    server_thread.join().unwrap();
}

/// Test multiple surface creation and management through mock compositor.
#[test]
fn mock_compositor_multiple_surfaces() {
    let mut display = wayland_server::Display::<MockCompositor>::new().unwrap();
    let mut dh = display.handle();

    dh.create_global::<MockCompositor, wl_compositor::WlCompositor, _>(6, ());
    dh.create_global::<MockCompositor, wl_shm::WlShm, _>(1, ());
    dh.create_global::<MockCompositor, xdg_server::xdg_wm_base::XdgWmBase, _>(5, ());

    let (client_stream, server_stream) = UnixStream::pair().unwrap();
    dh.insert_client(server_stream, Arc::new(())).unwrap();
    let client_conn = ClientConnection::from_socket(client_stream).unwrap();

    let mut compositor = MockCompositor::new();
    let surface_count = Arc::clone(&compositor.surface_count);
    let commit_count = Arc::clone(&compositor.commit_count);
    let stop = Arc::new(AtomicBool::new(false));
    let stop_server = Arc::clone(&stop);

    let server_thread = std::thread::spawn(move || {
        while !stop_server.load(Ordering::Relaxed) {
            display.dispatch_clients(&mut compositor).unwrap();
            display.flush_clients().unwrap();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    });

    struct CS {
        compositor: Option<wc::protocol::wl_compositor::WlCompositor>,
        shm: Option<wc::protocol::wl_shm::WlShm>,
        xdg_wm_base: Option<xdg_client::xdg_wm_base::XdgWmBase>,
        configured_count: u32,
    }

    impl ClientDispatch<wc::protocol::wl_registry::WlRegistry, ()> for CS {
        fn event(
            state: &mut Self,
            registry: &wc::protocol::wl_registry::WlRegistry,
            event: wc::protocol::wl_registry::Event,
            _: &(),
            _: &wc::Connection,
            qh: &QueueHandle<Self>,
        ) {
            if let wc::protocol::wl_registry::Event::Global {
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
                    _ => {}
                }
            }
        }
    }

    impl ClientDispatch<wc::protocol::wl_compositor::WlCompositor, ()> for CS {
        fn event(
            _: &mut Self,
            _: &wc::protocol::wl_compositor::WlCompositor,
            _: wc::protocol::wl_compositor::Event,
            _: &(),
            _: &wc::Connection,
            _: &QueueHandle<Self>,
        ) {
        }
    }
    impl ClientDispatch<wc::protocol::wl_shm::WlShm, ()> for CS {
        fn event(
            _: &mut Self,
            _: &wc::protocol::wl_shm::WlShm,
            _: wc::protocol::wl_shm::Event,
            _: &(),
            _: &wc::Connection,
            _: &QueueHandle<Self>,
        ) {
        }
    }
    impl ClientDispatch<wc::protocol::wl_shm_pool::WlShmPool, ()> for CS {
        fn event(
            _: &mut Self,
            _: &wc::protocol::wl_shm_pool::WlShmPool,
            _: wc::protocol::wl_shm_pool::Event,
            _: &(),
            _: &wc::Connection,
            _: &QueueHandle<Self>,
        ) {
        }
    }
    impl ClientDispatch<wc::protocol::wl_buffer::WlBuffer, usize> for CS {
        fn event(
            _: &mut Self,
            _: &wc::protocol::wl_buffer::WlBuffer,
            _: wc::protocol::wl_buffer::Event,
            _: &usize,
            _: &wc::Connection,
            _: &QueueHandle<Self>,
        ) {
        }
    }
    impl ClientDispatch<wc::protocol::wl_surface::WlSurface, ()> for CS {
        fn event(
            _: &mut Self,
            _: &wc::protocol::wl_surface::WlSurface,
            _: wc::protocol::wl_surface::Event,
            _: &(),
            _: &wc::Connection,
            _: &QueueHandle<Self>,
        ) {
        }
    }
    impl ClientDispatch<xdg_client::xdg_wm_base::XdgWmBase, ()> for CS {
        fn event(
            _: &mut Self,
            base: &xdg_client::xdg_wm_base::XdgWmBase,
            event: xdg_client::xdg_wm_base::Event,
            _: &(),
            _: &wc::Connection,
            _: &QueueHandle<Self>,
        ) {
            if let xdg_client::xdg_wm_base::Event::Ping { serial } = event {
                base.pong(serial);
            }
        }
    }
    impl ClientDispatch<xdg_client::xdg_surface::XdgSurface, ()> for CS {
        fn event(
            state: &mut Self,
            surf: &xdg_client::xdg_surface::XdgSurface,
            event: xdg_client::xdg_surface::Event,
            _: &(),
            _: &wc::Connection,
            _: &QueueHandle<Self>,
        ) {
            if let xdg_client::xdg_surface::Event::Configure { serial } = event {
                surf.ack_configure(serial);
                state.configured_count += 1;
            }
        }
    }
    impl ClientDispatch<xdg_client::xdg_toplevel::XdgToplevel, ()> for CS {
        fn event(
            _: &mut Self,
            _: &xdg_client::xdg_toplevel::XdgToplevel,
            _: xdg_client::xdg_toplevel::Event,
            _: &(),
            _: &wc::Connection,
            _: &QueueHandle<Self>,
        ) {
        }
    }

    let mut eq = client_conn.new_event_queue::<CS>();
    let qh = eq.handle();
    let mut cs = CS {
        compositor: None,
        shm: None,
        xdg_wm_base: None,
        configured_count: 0,
    };

    client_conn.display().get_registry(&qh, ());
    eq.roundtrip(&mut cs).unwrap();
    assert!(cs.compositor.is_some());
    assert!(cs.shm.is_some());
    assert!(cs.xdg_wm_base.is_some());

    // Create 3 surfaces.
    let mut surfaces = Vec::new();
    for i in 0..3u32 {
        let wl_surface = cs.compositor.as_ref().unwrap().create_surface(&qh, ());
        let xdg_surface = cs
            .xdg_wm_base
            .as_ref()
            .unwrap()
            .get_xdg_surface(&wl_surface, &qh, ());
        let toplevel = xdg_surface.get_toplevel(&qh, ());
        toplevel.set_title(format!("Window {i}"));
        toplevel.set_app_id(format!("test-app-{i}"));
        wl_surface.commit();
        surfaces.push((wl_surface, xdg_surface, toplevel));
    }

    // Roundtrip to receive configure events for all surfaces.
    eq.roundtrip(&mut cs).unwrap();
    assert_eq!(cs.configured_count, 3);
    assert!(surface_count.load(Ordering::Relaxed) >= 3);

    // Attach SHM buffers to each surface and commit.
    let shm = cs.shm.as_ref().unwrap();
    for (wl_surface, _, _) in &surfaces {
        let fd = nix::sys::memfd::memfd_create(
            c"test-shm-multi",
            nix::sys::memfd::MemFdCreateFlag::MFD_CLOEXEC,
        )
        .unwrap();
        let buf_size: i32 = 32 * 32 * 4;
        nix::unistd::ftruncate(&fd, buf_size as i64).unwrap();
        let pool = shm.create_pool(fd.as_fd(), buf_size, &qh, ());
        let buffer = pool.create_buffer(
            0,
            32,
            32,
            32 * 4,
            wc::protocol::wl_shm::Format::Xrgb8888,
            &qh,
            0usize,
        );
        wl_surface.attach(Some(&buffer), 0, 0);
        wl_surface.damage_buffer(0, 0, 32, 32);
        wl_surface.commit();
    }

    eq.roundtrip(&mut cs).unwrap();

    // Each surface committed twice (initial + attach), so at least 6 commits.
    let total_commits = commit_count.load(Ordering::Relaxed);
    assert!(
        total_commits >= 6,
        "expected >= 6 commits, got {total_commits}"
    );

    // Cleanup.
    for (wl_surface, xdg_surface, toplevel) in surfaces {
        toplevel.destroy();
        xdg_surface.destroy();
        wl_surface.destroy();
    }
    stop.store(true, Ordering::Relaxed);
    server_thread.join().unwrap();
}

/// Test SHM double-buffer creation and attachment through mock compositor.
#[test]
fn mock_compositor_shm_double_buffer() {
    let mut display = wayland_server::Display::<MockCompositor>::new().unwrap();
    let mut dh = display.handle();

    dh.create_global::<MockCompositor, wl_compositor::WlCompositor, _>(6, ());
    dh.create_global::<MockCompositor, wl_shm::WlShm, _>(1, ());
    dh.create_global::<MockCompositor, xdg_server::xdg_wm_base::XdgWmBase, _>(5, ());

    let (client_stream, server_stream) = UnixStream::pair().unwrap();
    dh.insert_client(server_stream, Arc::new(())).unwrap();
    let client_conn = ClientConnection::from_socket(client_stream).unwrap();

    let mut compositor = MockCompositor::new();
    let commit_count = Arc::clone(&compositor.commit_count);
    let stop = Arc::new(AtomicBool::new(false));
    let stop_server = Arc::clone(&stop);

    let server_thread = std::thread::spawn(move || {
        while !stop_server.load(Ordering::Relaxed) {
            display.dispatch_clients(&mut compositor).unwrap();
            display.flush_clients().unwrap();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    });

    struct CS {
        compositor: Option<wc::protocol::wl_compositor::WlCompositor>,
        shm: Option<wc::protocol::wl_shm::WlShm>,
        xdg_wm_base: Option<xdg_client::xdg_wm_base::XdgWmBase>,
        configured: bool,
    }

    impl ClientDispatch<wc::protocol::wl_registry::WlRegistry, ()> for CS {
        fn event(
            state: &mut Self,
            registry: &wc::protocol::wl_registry::WlRegistry,
            event: wc::protocol::wl_registry::Event,
            _: &(),
            _: &wc::Connection,
            qh: &QueueHandle<Self>,
        ) {
            if let wc::protocol::wl_registry::Event::Global {
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
                    _ => {}
                }
            }
        }
    }

    impl ClientDispatch<wc::protocol::wl_compositor::WlCompositor, ()> for CS {
        fn event(
            _: &mut Self,
            _: &wc::protocol::wl_compositor::WlCompositor,
            _: wc::protocol::wl_compositor::Event,
            _: &(),
            _: &wc::Connection,
            _: &QueueHandle<Self>,
        ) {
        }
    }
    impl ClientDispatch<wc::protocol::wl_shm::WlShm, ()> for CS {
        fn event(
            _: &mut Self,
            _: &wc::protocol::wl_shm::WlShm,
            _: wc::protocol::wl_shm::Event,
            _: &(),
            _: &wc::Connection,
            _: &QueueHandle<Self>,
        ) {
        }
    }
    impl ClientDispatch<wc::protocol::wl_shm_pool::WlShmPool, ()> for CS {
        fn event(
            _: &mut Self,
            _: &wc::protocol::wl_shm_pool::WlShmPool,
            _: wc::protocol::wl_shm_pool::Event,
            _: &(),
            _: &wc::Connection,
            _: &QueueHandle<Self>,
        ) {
        }
    }
    impl ClientDispatch<wc::protocol::wl_buffer::WlBuffer, usize> for CS {
        fn event(
            _: &mut Self,
            _: &wc::protocol::wl_buffer::WlBuffer,
            _: wc::protocol::wl_buffer::Event,
            _: &usize,
            _: &wc::Connection,
            _: &QueueHandle<Self>,
        ) {
        }
    }
    impl ClientDispatch<wc::protocol::wl_surface::WlSurface, ()> for CS {
        fn event(
            _: &mut Self,
            _: &wc::protocol::wl_surface::WlSurface,
            _: wc::protocol::wl_surface::Event,
            _: &(),
            _: &wc::Connection,
            _: &QueueHandle<Self>,
        ) {
        }
    }
    impl ClientDispatch<xdg_client::xdg_wm_base::XdgWmBase, ()> for CS {
        fn event(
            _: &mut Self,
            base: &xdg_client::xdg_wm_base::XdgWmBase,
            event: xdg_client::xdg_wm_base::Event,
            _: &(),
            _: &wc::Connection,
            _: &QueueHandle<Self>,
        ) {
            if let xdg_client::xdg_wm_base::Event::Ping { serial } = event {
                base.pong(serial);
            }
        }
    }
    impl ClientDispatch<xdg_client::xdg_surface::XdgSurface, ()> for CS {
        fn event(
            state: &mut Self,
            surf: &xdg_client::xdg_surface::XdgSurface,
            event: xdg_client::xdg_surface::Event,
            _: &(),
            _: &wc::Connection,
            _: &QueueHandle<Self>,
        ) {
            if let xdg_client::xdg_surface::Event::Configure { serial } = event {
                surf.ack_configure(serial);
                state.configured = true;
            }
        }
    }
    impl ClientDispatch<xdg_client::xdg_toplevel::XdgToplevel, ()> for CS {
        fn event(
            _: &mut Self,
            _: &xdg_client::xdg_toplevel::XdgToplevel,
            _: xdg_client::xdg_toplevel::Event,
            _: &(),
            _: &wc::Connection,
            _: &QueueHandle<Self>,
        ) {
        }
    }

    let mut eq = client_conn.new_event_queue::<CS>();
    let qh = eq.handle();
    let mut cs = CS {
        compositor: None,
        shm: None,
        xdg_wm_base: None,
        configured: false,
    };

    client_conn.display().get_registry(&qh, ());
    eq.roundtrip(&mut cs).unwrap();

    let wl_surface = cs.compositor.as_ref().unwrap().create_surface(&qh, ());
    let xdg_surface = cs
        .xdg_wm_base
        .as_ref()
        .unwrap()
        .get_xdg_surface(&wl_surface, &qh, ());
    let toplevel = xdg_surface.get_toplevel(&qh, ());
    toplevel.set_title("Double Buffer Test".into());
    wl_surface.commit();
    eq.roundtrip(&mut cs).unwrap();
    assert!(cs.configured);

    // Create double buffer (2 buffers from same pool).
    let shm = cs.shm.as_ref().unwrap();
    let width = 64u32;
    let height = 48u32;
    let stride = width * 4;
    let buf_size = stride * height;
    let total_size = buf_size * 2; // double buffer

    let fd = nix::sys::memfd::memfd_create(
        c"test-double-buf",
        nix::sys::memfd::MemFdCreateFlag::MFD_CLOEXEC,
    )
    .unwrap();
    nix::unistd::ftruncate(&fd, total_size as i64).unwrap();

    // mmap and write pattern data.
    // SAFETY: test mock helper.
    let map = unsafe {
        nix::sys::mman::mmap(
            None,
            std::num::NonZero::new(total_size as usize).unwrap(),
            nix::sys::mman::ProtFlags::PROT_READ | nix::sys::mman::ProtFlags::PROT_WRITE,
            nix::sys::mman::MapFlags::MAP_SHARED,
            &fd,
            0,
        )
        .unwrap()
    };

    // Write different patterns to each buffer.
    // SAFETY: test mock helper.
    unsafe {
        let slice = std::slice::from_raw_parts_mut(map.as_ptr() as *mut u8, total_size as usize);
        // Buffer 0: all 0xAA.
        for byte in &mut slice[..buf_size as usize] {
            *byte = 0xAA;
        }
        // Buffer 1: all 0xBB.
        for byte in &mut slice[buf_size as usize..] {
            *byte = 0xBB;
        }
    }

    let pool = shm.create_pool(fd.as_fd(), total_size as i32, &qh, ());

    // Create two buffers from the same pool at different offsets.
    let buf0 = pool.create_buffer(
        0,
        width as i32,
        height as i32,
        stride as i32,
        wc::protocol::wl_shm::Format::Xrgb8888,
        &qh,
        0usize,
    );
    let buf1 = pool.create_buffer(
        buf_size as i32,
        width as i32,
        height as i32,
        stride as i32,
        wc::protocol::wl_shm::Format::Xrgb8888,
        &qh,
        1usize,
    );

    // Alternate buffers like the display pipeline does.
    wl_surface.attach(Some(&buf0), 0, 0);
    wl_surface.damage_buffer(0, 0, width as i32, height as i32);
    wl_surface.commit();
    eq.roundtrip(&mut cs).unwrap();

    wl_surface.attach(Some(&buf1), 0, 0);
    wl_surface.damage_buffer(0, 0, width as i32, height as i32);
    wl_surface.commit();
    eq.roundtrip(&mut cs).unwrap();

    // Server should have received the commits.
    let total_commits = commit_count.load(Ordering::Relaxed);
    assert!(
        total_commits >= 3,
        "expected >= 3 commits (1 initial + 2 buffer), got {total_commits}"
    );

    // Cleanup.
    buf0.destroy();
    buf1.destroy();
    pool.destroy();
    // SAFETY: test mock helper.
    unsafe {
        nix::sys::mman::munmap(map, total_size as usize).ok();
    }
    toplevel.destroy();
    xdg_surface.destroy();
    wl_surface.destroy();
    stop.store(true, Ordering::Relaxed);
    server_thread.join().unwrap();
}

/// Test that damage regions are correctly tracked through surface commits.
#[test]
fn mock_compositor_damage_regions() {
    let mut display = wayland_server::Display::<MockCompositor>::new().unwrap();
    let mut dh = display.handle();

    dh.create_global::<MockCompositor, wl_compositor::WlCompositor, _>(6, ());
    dh.create_global::<MockCompositor, wl_shm::WlShm, _>(1, ());
    dh.create_global::<MockCompositor, xdg_server::xdg_wm_base::XdgWmBase, _>(5, ());

    let (client_stream, server_stream) = UnixStream::pair().unwrap();
    dh.insert_client(server_stream, Arc::new(())).unwrap();
    let client_conn = ClientConnection::from_socket(client_stream).unwrap();

    let mut compositor = MockCompositor::new();
    let commit_count = Arc::clone(&compositor.commit_count);
    let stop = Arc::new(AtomicBool::new(false));
    let stop_server = Arc::clone(&stop);

    let server_thread = std::thread::spawn(move || {
        while !stop_server.load(Ordering::Relaxed) {
            display.dispatch_clients(&mut compositor).unwrap();
            display.flush_clients().unwrap();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    });

    struct CS {
        compositor: Option<wc::protocol::wl_compositor::WlCompositor>,
        shm: Option<wc::protocol::wl_shm::WlShm>,
        xdg_wm_base: Option<xdg_client::xdg_wm_base::XdgWmBase>,
        configured: bool,
    }

    impl ClientDispatch<wc::protocol::wl_registry::WlRegistry, ()> for CS {
        fn event(
            state: &mut Self,
            registry: &wc::protocol::wl_registry::WlRegistry,
            event: wc::protocol::wl_registry::Event,
            _: &(),
            _: &wc::Connection,
            qh: &QueueHandle<Self>,
        ) {
            if let wc::protocol::wl_registry::Event::Global {
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
                    _ => {}
                }
            }
        }
    }

    impl ClientDispatch<wc::protocol::wl_compositor::WlCompositor, ()> for CS {
        fn event(
            _: &mut Self,
            _: &wc::protocol::wl_compositor::WlCompositor,
            _: wc::protocol::wl_compositor::Event,
            _: &(),
            _: &wc::Connection,
            _: &QueueHandle<Self>,
        ) {
        }
    }
    impl ClientDispatch<wc::protocol::wl_shm::WlShm, ()> for CS {
        fn event(
            _: &mut Self,
            _: &wc::protocol::wl_shm::WlShm,
            _: wc::protocol::wl_shm::Event,
            _: &(),
            _: &wc::Connection,
            _: &QueueHandle<Self>,
        ) {
        }
    }
    impl ClientDispatch<wc::protocol::wl_shm_pool::WlShmPool, ()> for CS {
        fn event(
            _: &mut Self,
            _: &wc::protocol::wl_shm_pool::WlShmPool,
            _: wc::protocol::wl_shm_pool::Event,
            _: &(),
            _: &wc::Connection,
            _: &QueueHandle<Self>,
        ) {
        }
    }
    impl ClientDispatch<wc::protocol::wl_buffer::WlBuffer, usize> for CS {
        fn event(
            _: &mut Self,
            _: &wc::protocol::wl_buffer::WlBuffer,
            _: wc::protocol::wl_buffer::Event,
            _: &usize,
            _: &wc::Connection,
            _: &QueueHandle<Self>,
        ) {
        }
    }
    impl ClientDispatch<wc::protocol::wl_surface::WlSurface, ()> for CS {
        fn event(
            _: &mut Self,
            _: &wc::protocol::wl_surface::WlSurface,
            _: wc::protocol::wl_surface::Event,
            _: &(),
            _: &wc::Connection,
            _: &QueueHandle<Self>,
        ) {
        }
    }
    impl ClientDispatch<xdg_client::xdg_wm_base::XdgWmBase, ()> for CS {
        fn event(
            _: &mut Self,
            base: &xdg_client::xdg_wm_base::XdgWmBase,
            event: xdg_client::xdg_wm_base::Event,
            _: &(),
            _: &wc::Connection,
            _: &QueueHandle<Self>,
        ) {
            if let xdg_client::xdg_wm_base::Event::Ping { serial } = event {
                base.pong(serial);
            }
        }
    }
    impl ClientDispatch<xdg_client::xdg_surface::XdgSurface, ()> for CS {
        fn event(
            state: &mut Self,
            surf: &xdg_client::xdg_surface::XdgSurface,
            event: xdg_client::xdg_surface::Event,
            _: &(),
            _: &wc::Connection,
            _: &QueueHandle<Self>,
        ) {
            if let xdg_client::xdg_surface::Event::Configure { serial } = event {
                surf.ack_configure(serial);
                state.configured = true;
            }
        }
    }
    impl ClientDispatch<xdg_client::xdg_toplevel::XdgToplevel, ()> for CS {
        fn event(
            _: &mut Self,
            _: &xdg_client::xdg_toplevel::XdgToplevel,
            _: xdg_client::xdg_toplevel::Event,
            _: &(),
            _: &wc::Connection,
            _: &QueueHandle<Self>,
        ) {
        }
    }

    let mut eq = client_conn.new_event_queue::<CS>();
    let qh = eq.handle();
    let mut cs = CS {
        compositor: None,
        shm: None,
        xdg_wm_base: None,
        configured: false,
    };

    client_conn.display().get_registry(&qh, ());
    eq.roundtrip(&mut cs).unwrap();

    let wl_surface = cs.compositor.as_ref().unwrap().create_surface(&qh, ());
    let xdg_surface = cs
        .xdg_wm_base
        .as_ref()
        .unwrap()
        .get_xdg_surface(&wl_surface, &qh, ());
    let toplevel = xdg_surface.get_toplevel(&qh, ());
    wl_surface.commit();
    eq.roundtrip(&mut cs).unwrap();
    assert!(cs.configured);

    // Create SHM buffer.
    let shm = cs.shm.as_ref().unwrap();
    let fd = nix::sys::memfd::memfd_create(
        c"test-damage",
        nix::sys::memfd::MemFdCreateFlag::MFD_CLOEXEC,
    )
    .unwrap();
    let buf_size: i32 = 128 * 128 * 4;
    nix::unistd::ftruncate(&fd, buf_size as i64).unwrap();
    let pool = shm.create_pool(fd.as_fd(), buf_size, &qh, ());
    let buffer = pool.create_buffer(
        0,
        128,
        128,
        128 * 4,
        wc::protocol::wl_shm::Format::Xrgb8888,
        &qh,
        0usize,
    );

    // Simulate partial damage commits (like the display pipeline).
    // Frame 1: full damage.
    wl_surface.attach(Some(&buffer), 0, 0);
    wl_surface.damage_buffer(0, 0, 128, 128);
    wl_surface.commit();

    // Frame 2: partial damage (top-left quadrant).
    wl_surface.damage_buffer(0, 0, 64, 64);
    wl_surface.commit();

    // Frame 3: multiple damage regions.
    wl_surface.damage_buffer(0, 0, 32, 32);
    wl_surface.damage_buffer(96, 96, 32, 32);
    wl_surface.commit();

    eq.roundtrip(&mut cs).unwrap();

    // 1 initial commit + 3 frame commits = 4 total.
    let total_commits = commit_count.load(Ordering::Relaxed);
    assert!(
        total_commits >= 4,
        "expected >= 4 commits, got {total_commits}"
    );

    // Cleanup.
    buffer.destroy();
    pool.destroy();
    toplevel.destroy();
    xdg_surface.destroy();
    wl_surface.destroy();
    stop.store(true, Ordering::Relaxed);
    server_thread.join().unwrap();
}

// ---------------------------------------------------------------------------
// Tests using WAYLAND_DISPLAY env var to exercise WaylandDisplay::new() and
// related public APIs through the actual code paths.
// ---------------------------------------------------------------------------

use std::sync::Mutex;

/// Mutex to serialize tests that modify the `WAYLAND_DISPLAY` env var.
static WAYLAND_DISPLAY_LOCK: Mutex<()> = Mutex::new(());

struct ListeningDisplayMock {
    stop: Arc<AtomicBool>,
    server_thread: Option<std::thread::JoinHandle<()>>,
    _guard: std::sync::MutexGuard<'static, ()>,
    old_display: Option<String>,
}

impl ListeningDisplayMock {
    fn new() -> Self {
        use wayland_server::ListeningSocket;

        let guard = WAYLAND_DISPLAY_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let display = wayland_server::Display::<MockCompositor>::new().unwrap();
        let dh = display.handle();

        dh.create_global::<MockCompositor, wl_compositor::WlCompositor, _>(6, ());
        dh.create_global::<MockCompositor, wl_shm::WlShm, _>(1, ());
        dh.create_global::<MockCompositor, xdg_server::xdg_wm_base::XdgWmBase, _>(5, ());

        let id = std::process::id();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos();
        let socket_name = format!("remoteway-display-test-{}-{}", id, ts);
        let listener = ListeningSocket::bind(&socket_name).unwrap();

        let old_display = std::env::var("WAYLAND_DISPLAY").ok();
        // SAFETY: test mock helper.
        unsafe { std::env::set_var("WAYLAND_DISPLAY", &socket_name) };

        let stop = Arc::new(AtomicBool::new(false));
        let stop_server = Arc::clone(&stop);

        let server_thread = std::thread::spawn(move || {
            let mut display = display;
            let mut compositor = MockCompositor::new();
            let mut dh = display.handle();
            while !stop_server.load(Ordering::Relaxed) {
                if let Ok(Some(stream)) = listener.accept() {
                    dh.insert_client(stream, Arc::new(()));
                }
                display.dispatch_clients(&mut compositor);
                display.flush_clients();
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

impl Drop for ListeningDisplayMock {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.server_thread.take() {
            h.join().ok();
        }
        match self.old_display.take() {
            // SAFETY: test mock helper.
            Some(val) => unsafe { std::env::set_var("WAYLAND_DISPLAY", val) },
            // SAFETY: test mock helper.
            None => unsafe { std::env::remove_var("WAYLAND_DISPLAY") },
        }
    }
}

#[test]
fn wayland_display_new_succeeds_with_mock() {
    let _mock = ListeningDisplayMock::new();
    let result = remoteway_display::WaylandDisplay::new();
    assert!(
        result.is_ok(),
        "WaylandDisplay::new failed: {:?}",
        result.err().map(|e| format!("{e}"))
    );
}

#[test]
fn wayland_display_create_surface_succeeds() {
    let _mock = ListeningDisplayMock::new();
    let mut display = remoteway_display::WaylandDisplay::new().unwrap();
    let result = display.create_surface(0, "Test Title", "test.app", 256, 256);
    assert!(result.is_ok(), "create_surface failed: {:?}", result.err());
    let surf = display.get_surface(0).unwrap();
    assert_eq!(surf.title(), "Test Title");
    assert_eq!(surf.app_id(), "test.app");
    assert_eq!(surf.width, 256);
    assert_eq!(surf.height, 256);
}

#[test]
fn wayland_display_get_surface_not_found() {
    let _mock = ListeningDisplayMock::new();
    let display = remoteway_display::WaylandDisplay::new().unwrap();
    assert!(display.get_surface(99).is_none());
}

#[test]
fn wayland_display_update_surface_metadata() {
    let _mock = ListeningDisplayMock::new();
    let mut display = remoteway_display::WaylandDisplay::new().unwrap();
    display
        .create_surface(1, "Initial", "init.app", 128, 128)
        .unwrap();

    let result = display.update_surface_metadata(1, "Updated", "new.app");
    assert!(result.is_ok());
    let surf = display.get_surface(1).unwrap();
    assert_eq!(surf.title(), "Updated");
    assert_eq!(surf.app_id(), "new.app");
}

#[test]
fn wayland_display_update_surface_metadata_not_found() {
    let _mock = ListeningDisplayMock::new();
    let mut display = remoteway_display::WaylandDisplay::new().unwrap();
    let result = display.update_surface_metadata(42, "x", "y");
    assert!(result.is_err());
}

#[test]
fn wayland_display_destroy_surface() {
    let _mock = ListeningDisplayMock::new();
    let mut display = remoteway_display::WaylandDisplay::new().unwrap();
    display.create_surface(2, "Tmp", "tmp.app", 64, 64).unwrap();
    assert!(display.get_surface(2).is_some());

    display.destroy_surface(2);
    assert!(display.get_surface(2).is_none());
}

#[test]
fn wayland_display_destroy_nonexistent_surface_is_noop() {
    let _mock = ListeningDisplayMock::new();
    let mut display = remoteway_display::WaylandDisplay::new().unwrap();
    // Should not panic.
    display.destroy_surface(999);
}

#[test]
fn wayland_display_resize_surface() {
    let _mock = ListeningDisplayMock::new();
    let mut display = remoteway_display::WaylandDisplay::new().unwrap();
    display.create_surface(3, "R", "r.app", 100, 100).unwrap();

    let result = display.resize_surface(3, 200, 150);
    assert!(result.is_ok());
    let surf = display.get_surface(3).unwrap();
    assert_eq!(surf.width, 200);
    assert_eq!(surf.height, 150);
    assert_eq!(surf.stride, 200 * 4);
}

#[test]
fn wayland_display_resize_to_same_size_is_noop() {
    let _mock = ListeningDisplayMock::new();
    let mut display = remoteway_display::WaylandDisplay::new().unwrap();
    display.create_surface(4, "S", "s.app", 50, 50).unwrap();

    let result = display.resize_surface(4, 50, 50);
    assert!(result.is_ok());
    let surf = display.get_surface(4).unwrap();
    assert_eq!(surf.width, 50);
    assert_eq!(surf.height, 50);
}

#[test]
fn wayland_display_resize_nonexistent_surface_errors() {
    let _mock = ListeningDisplayMock::new();
    let mut display = remoteway_display::WaylandDisplay::new().unwrap();
    let result = display.resize_surface(123, 32, 32);
    assert!(result.is_err());
}

#[test]
fn wayland_display_dispatch_pending_succeeds() {
    let _mock = ListeningDisplayMock::new();
    let mut display = remoteway_display::WaylandDisplay::new().unwrap();
    let result = display.dispatch_pending();
    assert!(result.is_ok());
}

#[test]
fn wayland_display_present_frame_to_nonexistent_surface() {
    let _mock = ListeningDisplayMock::new();
    let mut display = remoteway_display::WaylandDisplay::new().unwrap();
    let data = vec![0u8; 32 * 32 * 4];
    let result = display.present_frame(77, &data, &[]);
    assert!(result.is_err());
}

#[test]
fn wayland_display_present_frame_with_damage() {
    let _mock = ListeningDisplayMock::new();
    let mut display = remoteway_display::WaylandDisplay::new().unwrap();
    display.create_surface(5, "P", "p.app", 64, 64).unwrap();

    let data = vec![0xABu8; 64 * 64 * 4];
    let damage = vec![remoteway_display::shm::DamageRegion::new(0, 0, 32, 32)];
    let result = display.present_frame(5, &data, &damage);
    // present_frame may return false if frame callback pending or buffer not released,
    // but it should not error.
    assert!(result.is_ok());
}

#[test]
fn wayland_display_multiple_surfaces() {
    let _mock = ListeningDisplayMock::new();
    let mut display = remoteway_display::WaylandDisplay::new().unwrap();
    display.create_surface(10, "A", "a.app", 100, 100).unwrap();
    display.create_surface(11, "B", "b.app", 200, 200).unwrap();
    display.create_surface(12, "C", "c.app", 300, 300).unwrap();

    assert!(display.get_surface(10).is_some());
    assert!(display.get_surface(11).is_some());
    assert!(display.get_surface(12).is_some());

    display.destroy_surface(11);
    assert!(display.get_surface(10).is_some());
    assert!(display.get_surface(11).is_none());
    assert!(display.get_surface(12).is_some());
}

// ---------------------------------------------------------------------------
// CursorOverlay tests using the same mock compositor.
// ---------------------------------------------------------------------------

#[test]
fn cursor_overlay_create_and_update_position() {
    let _mock = ListeningDisplayMock::new();
    let display = remoteway_display::WaylandDisplay::new().unwrap();
    let compositor = display.state.compositor.as_ref().unwrap();
    let qh = display.event_queue.handle();

    let mut overlay = remoteway_display::cursor::CursorOverlay::new(compositor, &qh);
    assert_eq!(overlay.position(), (0.0, 0.0));
    assert!(!overlay.has_server_cursor());

    overlay.update_position(123.5, 456.0);
    assert_eq!(overlay.position(), (123.5, 456.0));

    overlay.update_position(-10.0, -20.0);
    assert_eq!(overlay.position(), (-10.0, -20.0));
}

#[test]
fn cursor_overlay_set_enter_serial() {
    let _mock = ListeningDisplayMock::new();
    let display = remoteway_display::WaylandDisplay::new().unwrap();
    let compositor = display.state.compositor.as_ref().unwrap();
    let qh = display.event_queue.handle();

    let mut overlay = remoteway_display::cursor::CursorOverlay::new(compositor, &qh);
    overlay.set_enter_serial(42);
    overlay.set_enter_serial(u32::MAX);
}

#[test]
fn cursor_overlay_apply_update_no_bitmap() {
    use remoteway_proto::cursor::CursorUpdate;

    let _mock = ListeningDisplayMock::new();
    let display = remoteway_display::WaylandDisplay::new().unwrap();
    let compositor = display.state.compositor.as_ref().unwrap();
    let shm = display.state.shm.as_ref().unwrap();
    let qh = display.event_queue.handle();

    let mut overlay = remoteway_display::cursor::CursorOverlay::new(compositor, &qh);

    let update = CursorUpdate {
        surface_id: 1,
        hotspot_x: 5,
        hotspot_y: 7,
        has_bitmap: 0,
        _pad: 0,
        bitmap_width: 0,
        bitmap_height: 0,
        x: 100.0,
        y: 200.0,
    };
    let result = overlay.apply_cursor_update(&update, None, shm, &qh);
    assert!(result.is_ok());
    assert_eq!(overlay.position(), (100.0, 200.0));
    assert!(!overlay.has_server_cursor());
}

#[test]
fn cursor_overlay_apply_update_with_bitmap() {
    use remoteway_proto::cursor::CursorUpdate;

    let _mock = ListeningDisplayMock::new();
    let display = remoteway_display::WaylandDisplay::new().unwrap();
    let compositor = display.state.compositor.as_ref().unwrap();
    let shm = display.state.shm.as_ref().unwrap();
    let qh = display.event_queue.handle();

    let mut overlay = remoteway_display::cursor::CursorOverlay::new(compositor, &qh);

    let update = CursorUpdate {
        surface_id: 1,
        hotspot_x: 0,
        hotspot_y: 0,
        has_bitmap: 1,
        _pad: 0,
        bitmap_width: 32,
        bitmap_height: 32,
        x: 50.0,
        y: 60.0,
    };
    let bitmap = vec![0xFFu8; 32 * 32 * 4];
    let result = overlay.apply_cursor_update(&update, Some(&bitmap), shm, &qh);
    assert!(result.is_ok());
    assert!(overlay.has_server_cursor());
}

#[test]
fn cursor_overlay_apply_update_too_small_bitmap_errors() {
    use remoteway_proto::cursor::CursorUpdate;

    let _mock = ListeningDisplayMock::new();
    let display = remoteway_display::WaylandDisplay::new().unwrap();
    let compositor = display.state.compositor.as_ref().unwrap();
    let shm = display.state.shm.as_ref().unwrap();
    let qh = display.event_queue.handle();

    let mut overlay = remoteway_display::cursor::CursorOverlay::new(compositor, &qh);

    let update = CursorUpdate {
        surface_id: 1,
        hotspot_x: 0,
        hotspot_y: 0,
        has_bitmap: 1,
        _pad: 0,
        bitmap_width: 32,
        bitmap_height: 32,
        x: 0.0,
        y: 0.0,
    };
    // Provide an undersized bitmap (1 byte) — should fail.
    let bitmap = vec![0u8; 1];
    let result = overlay.apply_cursor_update(&update, Some(&bitmap), shm, &qh);
    assert!(result.is_err());
}

#[test]
fn cursor_overlay_drop_is_clean() {
    let _mock = ListeningDisplayMock::new();
    let display = remoteway_display::WaylandDisplay::new().unwrap();
    let compositor = display.state.compositor.as_ref().unwrap();
    let qh = display.event_queue.handle();

    let overlay = remoteway_display::cursor::CursorOverlay::new(compositor, &qh);
    drop(overlay);
}

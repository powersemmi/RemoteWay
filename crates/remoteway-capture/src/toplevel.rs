use wayland_client::protocol::wl_registry;
use wayland_client::{Connection, Dispatch, QueueHandle, event_created_child};
use wayland_protocols_wlr::foreign_toplevel::v1::client::{
    zwlr_foreign_toplevel_handle_v1, zwlr_foreign_toplevel_manager_v1,
};

use crate::error::CaptureError;

/// Information about a toplevel (window) on the remote compositor.
#[derive(Debug, Clone)]
pub struct ToplevelInfo {
    pub title: String,
    pub app_id: String,
    pub activated: bool,
    pub minimized: bool,
}

/// Tracks toplevel windows via wlr-foreign-toplevel-management.
pub struct ToplevelTracker {
    toplevels: Vec<ToplevelInfo>,
}

impl ToplevelTracker {
    /// Perform a single round-trip to enumerate current toplevels.
    pub fn enumerate(conn: &Connection) -> Result<Self, CaptureError> {
        let display = conn.display();
        let mut event_queue = conn.new_event_queue::<ToplevelState>();
        let qh = event_queue.handle();

        let mut state = ToplevelState {
            toplevels: Vec::new(),
            manager: None,
            current: None,
        };

        display.get_registry(&qh, ());
        event_queue.roundtrip(&mut state)?;

        // If the manager was found, do another round-trip to get toplevel events.
        if state.manager.is_some() {
            event_queue.roundtrip(&mut state)?;
        }

        // Finalize last pending toplevel.
        if let Some(t) = state.current.take() {
            state.toplevels.push(t);
        }

        Ok(Self {
            toplevels: state.toplevels,
        })
    }

    pub fn toplevels(&self) -> &[ToplevelInfo] {
        &self.toplevels
    }

    pub fn find_by_app_id(&self, app_id: &str) -> Option<&ToplevelInfo> {
        self.toplevels.iter().find(|t| t.app_id == app_id)
    }
}

struct ToplevelState {
    toplevels: Vec<ToplevelInfo>,
    manager: Option<zwlr_foreign_toplevel_manager_v1::ZwlrForeignToplevelManagerV1>,
    current: Option<ToplevelInfo>,
}

impl Dispatch<wl_registry::WlRegistry, ()> for ToplevelState {
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
            && interface == "zwlr_foreign_toplevel_manager_v1"
        {
            state.manager = Some(registry.bind(name, version.min(3), qh, ()));
        }
    }
}

impl Dispatch<zwlr_foreign_toplevel_manager_v1::ZwlrForeignToplevelManagerV1, ()>
    for ToplevelState
{
    fn event(
        state: &mut Self,
        _proxy: &zwlr_foreign_toplevel_manager_v1::ZwlrForeignToplevelManagerV1,
        event: zwlr_foreign_toplevel_manager_v1::Event,
        _: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        if let zwlr_foreign_toplevel_manager_v1::Event::Finished = event {
            // Manager is done. Finalize.
            if let Some(t) = state.current.take() {
                state.toplevels.push(t);
            }
        }
    }

    event_created_child!(Self, zwlr_foreign_toplevel_manager_v1::ZwlrForeignToplevelManagerV1, [
        zwlr_foreign_toplevel_manager_v1::EVT_TOPLEVEL_OPCODE => (zwlr_foreign_toplevel_handle_v1::ZwlrForeignToplevelHandleV1, ()),
    ]);
}

impl Dispatch<zwlr_foreign_toplevel_handle_v1::ZwlrForeignToplevelHandleV1, ()> for ToplevelState {
    fn event(
        state: &mut Self,
        _proxy: &zwlr_foreign_toplevel_handle_v1::ZwlrForeignToplevelHandleV1,
        event: zwlr_foreign_toplevel_handle_v1::Event,
        _: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        match event {
            zwlr_foreign_toplevel_handle_v1::Event::Title { title } => {
                state
                    .current
                    .get_or_insert_with(|| ToplevelInfo {
                        title: String::new(),
                        app_id: String::new(),
                        activated: false,
                        minimized: false,
                    })
                    .title = title;
            }
            zwlr_foreign_toplevel_handle_v1::Event::AppId { app_id } => {
                if let Some(ref mut t) = state.current {
                    t.app_id = app_id;
                }
            }
            zwlr_foreign_toplevel_handle_v1::Event::State { state: st } => {
                if let Some(ref mut t) = state.current {
                    // State is a Vec<u8> of packed u32 enum values.
                    for chunk in st.chunks_exact(4) {
                        let val = u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        match val {
                            // Activated = 0, Minimized = 1 in the protocol.
                            0 => t.activated = true,
                            1 => t.minimized = true,
                            _ => {}
                        }
                    }
                }
            }
            zwlr_foreign_toplevel_handle_v1::Event::Done => {
                // Finalize this toplevel and start a new one.
                if let Some(t) = state.current.take() {
                    state.toplevels.push(t);
                }
            }
            zwlr_foreign_toplevel_handle_v1::Event::Closed => {
                // Drop the current without adding to list.
                state.current = None;
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toplevel_info_clone() {
        let info = ToplevelInfo {
            title: "Firefox".into(),
            app_id: "org.mozilla.firefox".into(),
            activated: true,
            minimized: false,
        };
        let cloned = info.clone();
        assert_eq!(cloned.title, "Firefox");
        assert!(cloned.activated);
    }

    #[test]
    fn find_by_app_id_in_list() {
        let tracker = ToplevelTracker {
            toplevels: vec![
                ToplevelInfo {
                    title: "Terminal".into(),
                    app_id: "org.gnome.Terminal".into(),
                    activated: false,
                    minimized: false,
                },
                ToplevelInfo {
                    title: "Firefox".into(),
                    app_id: "org.mozilla.firefox".into(),
                    activated: true,
                    minimized: false,
                },
            ],
        };
        assert!(tracker.find_by_app_id("org.mozilla.firefox").is_some());
        assert!(tracker.find_by_app_id("com.example.none").is_none());
    }

    #[test]
    fn toplevel_info_debug() {
        let info = ToplevelInfo {
            title: "VLC".into(),
            app_id: "org.videolan.VLC".into(),
            activated: false,
            minimized: true,
        };
        let dbg = format!("{:?}", info);
        assert!(dbg.contains("VLC"));
        assert!(dbg.contains("org.videolan.VLC"));
        assert!(dbg.contains("minimized: true"));
    }

    #[test]
    fn toplevel_info_minimized_flag() {
        let info = ToplevelInfo {
            title: "App".into(),
            app_id: "com.example.app".into(),
            activated: false,
            minimized: true,
        };
        assert!(info.minimized);
        assert!(!info.activated);
    }

    #[test]
    fn toplevels_returns_all() {
        let tracker = ToplevelTracker {
            toplevels: vec![
                ToplevelInfo {
                    title: "A".into(),
                    app_id: "a".into(),
                    activated: false,
                    minimized: false,
                },
                ToplevelInfo {
                    title: "B".into(),
                    app_id: "b".into(),
                    activated: true,
                    minimized: false,
                },
                ToplevelInfo {
                    title: "C".into(),
                    app_id: "c".into(),
                    activated: false,
                    minimized: true,
                },
            ],
        };
        assert_eq!(tracker.toplevels().len(), 3);
        assert_eq!(tracker.toplevels()[0].app_id, "a");
        assert_eq!(tracker.toplevels()[1].title, "B");
        assert!(tracker.toplevels()[2].minimized);
    }

    #[test]
    fn toplevels_empty_tracker() {
        let tracker = ToplevelTracker {
            toplevels: Vec::new(),
        };
        assert!(tracker.toplevels().is_empty());
        assert!(tracker.find_by_app_id("anything").is_none());
    }

    #[test]
    fn find_by_app_id_returns_first_match() {
        let tracker = ToplevelTracker {
            toplevels: vec![
                ToplevelInfo {
                    title: "First".into(),
                    app_id: "dup".into(),
                    activated: false,
                    minimized: false,
                },
                ToplevelInfo {
                    title: "Second".into(),
                    app_id: "dup".into(),
                    activated: true,
                    minimized: false,
                },
            ],
        };
        let found = tracker.find_by_app_id("dup").unwrap();
        assert_eq!(found.title, "First");
    }

    #[test]
    fn connection_fails_without_wayland() {
        if std::env::var("WAYLAND_DISPLAY").is_ok() {
            return;
        }
        // Without WAYLAND_DISPLAY, connect_to_env should fail,
        // which means ToplevelTracker::enumerate would also fail.
        let conn_result = wayland_client::Connection::connect_to_env();
        assert!(conn_result.is_err());
    }
}

use wayland_client::protocol::wl_output;
use wayland_client::protocol::wl_registry;
use wayland_client::{Connection, Dispatch, QueueHandle, WEnum};
use wayland_protocols::xdg::xdg_output::zv1::client::zxdg_output_manager_v1;
use wayland_protocols::xdg::xdg_output::zv1::client::zxdg_output_v1;

use crate::error::CaptureError;

/// Information about a connected display output.
#[derive(Debug, Clone)]
pub struct OutputInfo {
    pub name: String,
    pub description: String,
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
    pub refresh_mhz: i32,
    pub scale: i32,
    /// Wayland global name for this output.
    pub global_name: u32,
}

impl OutputInfo {
    fn new(global_name: u32) -> Self {
        Self {
            name: String::new(),
            description: String::new(),
            x: 0,
            y: 0,
            width: 0,
            height: 0,
            refresh_mhz: 0,
            scale: 1,
            global_name,
        }
    }
}

/// Enumerates Wayland outputs via wl_output + xdg-output-v1.
pub struct OutputEnumerator {
    outputs: Vec<OutputInfo>,
}

impl OutputEnumerator {
    /// Enumerate outputs from an existing Wayland connection.
    ///
    /// Performs a blocking round-trip to collect all output information.
    pub fn enumerate(conn: &Connection) -> Result<Self, CaptureError> {
        let display = conn.display();
        let mut event_queue = conn.new_event_queue::<OutputState>();
        let qh = event_queue.handle();

        let mut state = OutputState {
            outputs: Vec::new(),
            xdg_output_manager: None,
            pending_xdg_outputs: Vec::new(),
        };

        // Get the registry.
        display.get_registry(&qh, ());

        // First round-trip: discover globals (wl_output, xdg_output_manager).
        event_queue.roundtrip(&mut state)?;

        // If xdg_output_manager is available, request xdg_output for each wl_output.
        if let Some(ref manager) = state.xdg_output_manager {
            for (_info, wl_out) in &state.pending_xdg_outputs {
                manager.get_xdg_output(wl_out, &qh, ());
            }
            // Second round-trip: receive xdg_output name/description/logical_position.
            event_queue.roundtrip(&mut state)?;
        }

        Ok(Self {
            outputs: state.outputs,
        })
    }

    /// All discovered outputs.
    pub fn outputs(&self) -> &[OutputInfo] {
        &self.outputs
    }

    /// Find an output by its name (e.g., "HDMI-A-1", "eDP-1").
    pub fn find_by_name(&self, name: &str) -> Option<&OutputInfo> {
        self.outputs.iter().find(|o| o.name == name)
    }
}

struct OutputState {
    outputs: Vec<OutputInfo>,
    xdg_output_manager: Option<zxdg_output_manager_v1::ZxdgOutputManagerV1>,
    pending_xdg_outputs: Vec<(usize, wl_output::WlOutput)>,
}

impl Dispatch<wl_registry::WlRegistry, ()> for OutputState {
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
                "wl_output" => {
                    let wl_out: wl_output::WlOutput = registry.bind(name, version.min(4), qh, name);
                    let idx = state.outputs.len();
                    state.outputs.push(OutputInfo::new(name));
                    state.pending_xdg_outputs.push((idx, wl_out));
                }
                "zxdg_output_manager_v1" => {
                    state.xdg_output_manager = Some(registry.bind(name, version.min(3), qh, ()));
                }
                _ => {}
            }
        }
    }
}

impl Dispatch<wl_output::WlOutput, u32> for OutputState {
    fn event(
        state: &mut Self,
        _proxy: &wl_output::WlOutput,
        event: wl_output::Event,
        global_name: &u32,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        let Some(info) = state
            .outputs
            .iter_mut()
            .find(|o| o.global_name == *global_name)
        else {
            return;
        };
        match event {
            wl_output::Event::Geometry { x, y, .. } => {
                info.x = x;
                info.y = y;
            }
            wl_output::Event::Mode {
                flags,
                width,
                height,
                refresh,
            } => {
                // Only care about the current mode.
                let is_current = match flags {
                    WEnum::Value(f) => f.contains(wl_output::Mode::Current),
                    _ => false,
                };
                if is_current {
                    info.width = width;
                    info.height = height;
                    info.refresh_mhz = refresh;
                }
            }
            wl_output::Event::Scale { factor } => {
                info.scale = factor;
            }
            wl_output::Event::Name { name } => {
                info.name = name;
            }
            wl_output::Event::Description { description } => {
                info.description = description;
            }
            _ => {}
        }
    }
}

impl Dispatch<zxdg_output_manager_v1::ZxdgOutputManagerV1, ()> for OutputState {
    fn event(
        _state: &mut Self,
        _proxy: &zxdg_output_manager_v1::ZxdgOutputManagerV1,
        _event: zxdg_output_manager_v1::Event,
        _: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        // No events to handle.
    }
}

impl Dispatch<zxdg_output_v1::ZxdgOutputV1, ()> for OutputState {
    fn event(
        state: &mut Self,
        _proxy: &zxdg_output_v1::ZxdgOutputV1,
        event: zxdg_output_v1::Event,
        _: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        match event {
            zxdg_output_v1::Event::Name { name } => {
                // xdg-output name overrides wl_output name.
                if let Some(info) = state.outputs.last_mut() {
                    info.name = name;
                }
            }
            zxdg_output_v1::Event::Description { description } => {
                if let Some(info) = state.outputs.last_mut() {
                    info.description = description;
                }
            }
            zxdg_output_v1::Event::LogicalPosition { x, y } => {
                if let Some(info) = state.outputs.last_mut() {
                    info.x = x;
                    info.y = y;
                }
            }
            zxdg_output_v1::Event::LogicalSize { width, height } => {
                if let Some(info) = state.outputs.last_mut() {
                    info.width = width;
                    info.height = height;
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_info_new_defaults() {
        let info = OutputInfo::new(42);
        assert_eq!(info.global_name, 42);
        assert!(info.name.is_empty());
        assert_eq!(info.scale, 1);
    }

    #[test]
    fn output_info_clone() {
        let info = OutputInfo {
            name: "eDP-1".into(),
            description: "Built-in display".into(),
            x: 0,
            y: 0,
            width: 1920,
            height: 1080,
            refresh_mhz: 60000,
            scale: 2,
            global_name: 1,
        };
        let cloned = info.clone();
        assert_eq!(cloned.name, "eDP-1");
        assert_eq!(cloned.width, 1920);
    }

    #[test]
    fn find_by_name_in_list() {
        let enumerator = OutputEnumerator {
            outputs: vec![
                OutputInfo {
                    name: "HDMI-A-1".into(),
                    description: String::new(),
                    x: 0,
                    y: 0,
                    width: 3840,
                    height: 2160,
                    refresh_mhz: 60000,
                    scale: 1,
                    global_name: 1,
                },
                OutputInfo {
                    name: "eDP-1".into(),
                    description: String::new(),
                    x: 3840,
                    y: 0,
                    width: 1920,
                    height: 1080,
                    refresh_mhz: 60000,
                    scale: 1,
                    global_name: 2,
                },
            ],
        };
        assert!(enumerator.find_by_name("eDP-1").is_some());
        assert_eq!(enumerator.find_by_name("eDP-1").unwrap().width, 1920);
        assert!(enumerator.find_by_name("DP-3").is_none());
    }

    #[test]
    fn output_info_debug() {
        let info = OutputInfo::new(7);
        let dbg = format!("{:?}", info);
        assert!(dbg.contains("global_name: 7"));
    }

    #[test]
    fn outputs_returns_all() {
        let enumerator = OutputEnumerator {
            outputs: vec![OutputInfo::new(1), OutputInfo::new(2), OutputInfo::new(3)],
        };
        assert_eq!(enumerator.outputs().len(), 3);
        assert_eq!(enumerator.outputs()[0].global_name, 1);
        assert_eq!(enumerator.outputs()[2].global_name, 3);
    }

    #[test]
    fn outputs_empty_enumerator() {
        let enumerator = OutputEnumerator {
            outputs: Vec::new(),
        };
        assert!(enumerator.outputs().is_empty());
        assert!(enumerator.find_by_name("anything").is_none());
    }

    #[test]
    fn output_info_full_fields() {
        let info = OutputInfo {
            name: "DP-2".into(),
            description: "External monitor".into(),
            x: 1920,
            y: 0,
            width: 2560,
            height: 1440,
            refresh_mhz: 144000,
            scale: 2,
            global_name: 5,
        };
        assert_eq!(info.name, "DP-2");
        assert_eq!(info.description, "External monitor");
        assert_eq!(info.x, 1920);
        assert_eq!(info.y, 0);
        assert_eq!(info.width, 2560);
        assert_eq!(info.height, 1440);
        assert_eq!(info.refresh_mhz, 144000);
        assert_eq!(info.scale, 2);
        assert_eq!(info.global_name, 5);
    }

    #[test]
    fn enumerate_without_wayland_returns_error() {
        if std::env::var("WAYLAND_DISPLAY").is_ok() {
            return;
        }
        let result = wayland_client::Connection::connect_to_env();
        assert!(result.is_err());
    }
}

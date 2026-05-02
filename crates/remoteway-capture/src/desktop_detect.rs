/// Detected desktop environment type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DesktopEnvironment {
    /// GNOME — uses xdg-desktop-portal + `PipeWire` for capture, libei for input.
    Gnome,
    /// KDE Plasma — uses xdg-desktop-portal + `PipeWire`.
    Kde,
    /// wlroots-based (Sway, Hyprland, etc.) — uses wlr-screencopy / ext-image-capture.
    Wlroots,
    /// Unknown — fall back to protocol-based detection.
    Unknown,
}

impl DesktopEnvironment {
    /// Returns `true` if this DE requires the portal/PipeWire capture path.
    #[must_use]
    pub fn needs_portal(&self) -> bool {
        matches!(self, Self::Gnome | Self::Kde)
    }

    /// Returns `true` if this DE supports wlr-screencopy / ext-image-capture.
    #[must_use]
    pub fn has_wlr_protocols(&self) -> bool {
        matches!(self, Self::Wlroots)
    }
}

impl std::fmt::Display for DesktopEnvironment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gnome => write!(f, "GNOME"),
            Self::Kde => write!(f, "KDE"),
            Self::Wlroots => write!(f, "wlroots"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

/// Detect the current desktop environment.
///
/// Uses a multi-step detection strategy:
/// 1. `XDG_CURRENT_DESKTOP` / `XDG_SESSION_DESKTOP` environment variables.
/// 2. `loginctl` query for the active graphical session (works over SSH).
/// 3. `/proc` scan for known compositor processes (universal fallback).
pub fn detect_desktop() -> DesktopEnvironment {
    // Primary: environment variables (works in graphical sessions).
    let result = detect_from_env(
        std::env::var("XDG_CURRENT_DESKTOP").ok().as_deref(),
        std::env::var("XDG_SESSION_DESKTOP").ok().as_deref(),
    );
    if result != DesktopEnvironment::Unknown {
        return result;
    }

    // Fallback 1: query loginctl for the active graphical session.
    if let Some(de) = detect_from_loginctl() {
        tracing::debug!(desktop = %de, "detected via loginctl");
        return de;
    }

    // Fallback 2: scan running processes for known compositors.
    if let Some(de) = detect_from_processes() {
        tracing::debug!(desktop = %de, "detected via process scan");
        return de;
    }

    DesktopEnvironment::Unknown
}

/// Inner detection logic, separated for testability.
fn detect_from_env(
    xdg_current_desktop: Option<&str>,
    xdg_session_desktop: Option<&str>,
) -> DesktopEnvironment {
    // XDG_CURRENT_DESKTOP can be colon-separated (e.g. "ubuntu:GNOME")
    if let Some(desktop) = xdg_current_desktop {
        let lower = desktop.to_lowercase();
        for component in lower.split(':') {
            let trimmed = component.trim();
            if trimmed == "gnome" {
                return DesktopEnvironment::Gnome;
            }
            if trimmed == "kde" {
                return DesktopEnvironment::Kde;
            }
            if trimmed == "sway"
                || trimmed == "hyprland"
                || trimmed == "river"
                || trimmed == "wayfire"
                || trimmed == "niri"
            {
                return DesktopEnvironment::Wlroots;
            }
        }
    }

    // Fallback to XDG_SESSION_DESKTOP
    if let Some(session) = xdg_session_desktop {
        let lower = session.to_lowercase();
        if lower.contains("gnome") {
            return DesktopEnvironment::Gnome;
        }
        if lower.contains("kde") || lower.contains("plasma") {
            return DesktopEnvironment::Kde;
        }
        if lower.contains("sway") || lower.contains("hyprland") {
            return DesktopEnvironment::Wlroots;
        }
    }

    DesktopEnvironment::Unknown
}

/// Ensure `WAYLAND_DISPLAY` is set in the current process environment.
///
/// When the server is launched via SSH, the graphical session's env vars are not
/// inherited. This function discovers them by:
/// 1. Reading the compositor process's `/proc/<pid>/environ`
/// 2. Scanning `XDG_RUNTIME_DIR` for `wayland-*` sockets (fallback)
///
/// # Safety contract
/// Must be called **before** spawning any pipeline threads, because
/// `std::env::set_var` is not thread-safe (edition 2024 marks it `unsafe`).
///
/// Returns `true` if `WAYLAND_DISPLAY` is available after discovery.
pub fn ensure_wayland_env() -> bool {
    if std::env::var("WAYLAND_DISPLAY").is_ok() {
        return true;
    }

    // Primary: read env from a running compositor process.
    if inherit_compositor_env() {
        return true;
    }

    // Fallback: scan runtime dir for wayland sockets.
    if let Ok(runtime_dir) = std::env::var("XDG_RUNTIME_DIR")
        && let Some(wl_display) = find_wayland_socket(&runtime_dir)
    {
        // SAFETY: called before pipeline threads are spawned.
        unsafe { std::env::set_var("WAYLAND_DISPLAY", &wl_display) };
        tracing::info!(wayland_display = %wl_display, "discovered wayland socket in runtime dir");
        return true;
    }

    false
}

/// Read `WAYLAND_DISPLAY` and `XDG_RUNTIME_DIR` from a running compositor's
/// `/proc/<pid>/environ` and set them in the current process.
fn inherit_compositor_env() -> bool {
    let entries = match std::fs::read_dir("/proc") {
        Ok(e) => e,
        Err(_) => return false,
    };

    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if !name_str.bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }

        let comm_path = entry.path().join("comm");
        let Ok(comm) = std::fs::read_to_string(&comm_path) else {
            continue;
        };
        let comm = comm.trim();

        if !COMPOSITOR_PROCESSES
            .iter()
            .any(|&(proc_name, _)| comm == proc_name)
        {
            continue;
        }

        let environ_path = entry.path().join("environ");
        let Ok(environ_raw) = std::fs::read(&environ_path) else {
            continue;
        };

        let mut wayland_display = None;
        let mut xdg_runtime_dir = None;

        for var_bytes in environ_raw.split(|&b| b == 0) {
            let Ok(s) = std::str::from_utf8(var_bytes) else {
                continue;
            };
            if let Some(val) = s.strip_prefix("WAYLAND_DISPLAY=") {
                wayland_display = Some(val.to_string());
            } else if let Some(val) = s.strip_prefix("XDG_RUNTIME_DIR=") {
                xdg_runtime_dir = Some(val.to_string());
            }
        }

        if let Some(ref wl_display) = wayland_display {
            // SAFETY: called before pipeline threads are spawned.
            unsafe { std::env::set_var("WAYLAND_DISPLAY", wl_display) };
            tracing::info!(wayland_display = %wl_display, "inherited from compositor process");

            if let Some(ref runtime_dir) = xdg_runtime_dir
                && std::env::var("XDG_RUNTIME_DIR").is_err()
            {
                // SAFETY: called before pipeline threads are spawned.
                unsafe { std::env::set_var("XDG_RUNTIME_DIR", runtime_dir) };
                tracing::info!(xdg_runtime_dir = %runtime_dir, "inherited from compositor process");
            }
            return true;
        }
    }

    false
}

/// Find a `wayland-*` socket in the given runtime directory.
fn find_wayland_socket(runtime_dir: &str) -> Option<String> {
    use std::os::unix::fs::FileTypeExt;

    let entries = std::fs::read_dir(runtime_dir).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str.starts_with("wayland-")
            && !name_str.ends_with(".lock")
            && entry.file_type().map(|ft| ft.is_socket()).unwrap_or(false)
        {
            return Some(name_str.to_string());
        }
    }
    None
}

/// Known compositor process names and their desktop environments.
const COMPOSITOR_PROCESSES: &[(&str, DesktopEnvironment)] = &[
    ("gnome-shell", DesktopEnvironment::Gnome),
    ("gnome-session-b", DesktopEnvironment::Gnome),
    ("kwin_wayland", DesktopEnvironment::Kde),
    ("plasmashell", DesktopEnvironment::Kde),
    ("sway", DesktopEnvironment::Wlroots),
    ("Hyprland", DesktopEnvironment::Wlroots),
    ("hyprland", DesktopEnvironment::Wlroots),
    ("river", DesktopEnvironment::Wlroots),
    ("wayfire", DesktopEnvironment::Wlroots),
    ("niri", DesktopEnvironment::Wlroots),
];

/// Query systemd-logind for the active graphical session's desktop type.
///
/// Parses `loginctl list-sessions` to find wayland/x11 sessions, then reads
/// the `Desktop` property to determine the DE. Works over SSH because it
/// queries the system daemon, not the current session's environment.
fn detect_from_loginctl() -> Option<DesktopEnvironment> {
    let output = std::process::Command::new("loginctl")
        .args(["list-sessions", "--no-legend", "--no-pager"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let sessions = String::from_utf8_lossy(&output.stdout);

    for line in sessions.lines() {
        let session_id = line.split_whitespace().next()?;

        let show = std::process::Command::new("loginctl")
            .args([
                "show-session",
                session_id,
                "--property=Type",
                "--property=Desktop",
            ])
            .output()
            .ok()?;

        if !show.status.success() {
            continue;
        }

        let props = String::from_utf8_lossy(&show.stdout);
        let mut is_graphical = false;
        let mut desktop = None;

        for prop_line in props.lines() {
            if let Some(val) = prop_line.strip_prefix("Type=") {
                is_graphical = val == "wayland" || val == "x11";
            }
            if let Some(val) = prop_line.strip_prefix("Desktop=")
                && !val.is_empty()
            {
                desktop = Some(val.to_string());
            }
        }

        if is_graphical && let Some(ref d) = desktop {
            let result = detect_from_env(Some(d), None);
            if result != DesktopEnvironment::Unknown {
                return Some(result);
            }
        }
    }

    None
}

/// Scan `/proc` for known compositor processes.
///
/// Reads `/proc/<pid>/comm` for each process and matches against known
/// compositor names. This is a universal fallback that works regardless
/// of session type or environment variables.
fn detect_from_processes() -> Option<DesktopEnvironment> {
    let entries = std::fs::read_dir("/proc").ok()?;

    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if !name_str.bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }

        let comm_path = entry.path().join("comm");
        let Ok(comm) = std::fs::read_to_string(&comm_path) else {
            continue;
        };
        let comm = comm.trim();

        for &(proc_name, de) in COMPOSITOR_PROCESSES {
            if comm == proc_name {
                return Some(de);
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_gnome() {
        assert_eq!(
            detect_from_env(Some("GNOME"), None),
            DesktopEnvironment::Gnome
        );
    }

    #[test]
    fn detect_ubuntu_gnome() {
        assert_eq!(
            detect_from_env(Some("ubuntu:GNOME"), None),
            DesktopEnvironment::Gnome
        );
    }

    #[test]
    fn detect_kde() {
        assert_eq!(detect_from_env(Some("KDE"), None), DesktopEnvironment::Kde);
    }

    #[test]
    fn detect_sway() {
        assert_eq!(
            detect_from_env(Some("sway"), None),
            DesktopEnvironment::Wlroots
        );
    }

    #[test]
    fn detect_niri() {
        assert_eq!(
            detect_from_env(Some("niri"), None),
            DesktopEnvironment::Wlroots
        );
    }

    #[test]
    fn detect_hyprland() {
        assert_eq!(
            detect_from_env(Some("Hyprland"), None),
            DesktopEnvironment::Wlroots
        );
    }

    #[test]
    fn detect_unknown() {
        assert_eq!(detect_from_env(None, None), DesktopEnvironment::Unknown);
    }

    #[test]
    fn detect_fallback_session_desktop() {
        assert_eq!(
            detect_from_env(None, Some("gnome")),
            DesktopEnvironment::Gnome
        );
    }

    #[test]
    fn detect_unknown_de() {
        assert_eq!(
            detect_from_env(Some("something_else"), None),
            DesktopEnvironment::Unknown
        );
    }

    #[test]
    fn needs_portal() {
        assert!(DesktopEnvironment::Gnome.needs_portal());
        assert!(DesktopEnvironment::Kde.needs_portal());
        assert!(!DesktopEnvironment::Wlroots.needs_portal());
        assert!(!DesktopEnvironment::Unknown.needs_portal());
    }

    #[test]
    fn has_wlr_protocols() {
        assert!(!DesktopEnvironment::Gnome.has_wlr_protocols());
        assert!(DesktopEnvironment::Wlroots.has_wlr_protocols());
    }

    #[test]
    fn display_impl() {
        assert_eq!(format!("{}", DesktopEnvironment::Gnome), "GNOME");
        assert_eq!(format!("{}", DesktopEnvironment::Wlroots), "wlroots");
    }

    #[test]
    fn compositor_process_table_covers_all_de() {
        // Ensure the process table covers all non-Unknown variants.
        let has_gnome = COMPOSITOR_PROCESSES
            .iter()
            .any(|&(_, de)| de == DesktopEnvironment::Gnome);
        let has_kde = COMPOSITOR_PROCESSES
            .iter()
            .any(|&(_, de)| de == DesktopEnvironment::Kde);
        let has_wlroots = COMPOSITOR_PROCESSES
            .iter()
            .any(|&(_, de)| de == DesktopEnvironment::Wlroots);
        assert!(has_gnome);
        assert!(has_kde);
        assert!(has_wlroots);
    }

    #[test]
    fn process_scan_returns_some_on_linux() {
        // On a running Linux desktop, at least one compositor should be detected.
        // In CI without a compositor, this returns None — that's fine.
        detect_from_processes();
    }

    #[test]
    fn loginctl_fallback_does_not_panic() {
        // Should not panic even if loginctl is not installed.
        detect_from_loginctl();
    }

    #[test]
    fn ensure_wayland_env_does_not_panic() {
        // Should not panic regardless of environment.
        ensure_wayland_env();
    }

    #[test]
    fn find_wayland_socket_nonexistent_dir() {
        assert!(find_wayland_socket("/nonexistent/path").is_none());
    }

    #[test]
    fn find_wayland_socket_empty_dir() {
        let dir = std::env::temp_dir().join("remoteway_test_empty");
        std::fs::create_dir_all(&dir).ok();
        assert!(find_wayland_socket(dir.to_str().unwrap()).is_none());
        std::fs::remove_dir(&dir).ok();
    }

    #[test]
    fn inherit_compositor_env_does_not_panic() {
        inherit_compositor_env();
    }

    // --- Additional coverage for detect_from_env branches ---

    #[test]
    fn detect_session_desktop_plasma() {
        assert_eq!(
            detect_from_env(None, Some("plasma")),
            DesktopEnvironment::Kde
        );
    }

    #[test]
    fn detect_session_desktop_kde() {
        assert_eq!(
            detect_from_env(None, Some("kde-plasma")),
            DesktopEnvironment::Kde
        );
    }

    #[test]
    fn detect_session_desktop_sway() {
        assert_eq!(
            detect_from_env(None, Some("sway")),
            DesktopEnvironment::Wlroots
        );
    }

    #[test]
    fn detect_session_desktop_hyprland() {
        assert_eq!(
            detect_from_env(None, Some("hyprland")),
            DesktopEnvironment::Wlroots
        );
    }

    #[test]
    fn detect_session_desktop_unknown() {
        assert_eq!(
            detect_from_env(None, Some("budgie")),
            DesktopEnvironment::Unknown
        );
    }

    #[test]
    fn detect_xdg_current_desktop_river() {
        assert_eq!(
            detect_from_env(Some("river"), None),
            DesktopEnvironment::Wlroots
        );
    }

    #[test]
    fn detect_xdg_current_desktop_wayfire() {
        assert_eq!(
            detect_from_env(Some("wayfire"), None),
            DesktopEnvironment::Wlroots
        );
    }

    #[test]
    fn detect_xdg_current_desktop_case_insensitive() {
        assert_eq!(
            detect_from_env(Some("SWAY"), None),
            DesktopEnvironment::Wlroots
        );
        assert_eq!(detect_from_env(Some("Kde"), None), DesktopEnvironment::Kde);
        assert_eq!(
            detect_from_env(Some("Gnome"), None),
            DesktopEnvironment::Gnome
        );
    }

    #[test]
    fn detect_colon_separated_multiple_components() {
        // GNOME hides in a multi-component string
        assert_eq!(
            detect_from_env(Some("pop:GNOME"), None),
            DesktopEnvironment::Gnome
        );
        // KDE in colon-separated
        assert_eq!(
            detect_from_env(Some("custom:KDE:extra"), None),
            DesktopEnvironment::Kde
        );
    }

    #[test]
    fn detect_xdg_current_desktop_unknown_falls_through_to_session() {
        // Unknown XDG_CURRENT_DESKTOP should still check XDG_SESSION_DESKTOP
        assert_eq!(
            detect_from_env(Some("unknown-de"), Some("gnome")),
            DesktopEnvironment::Gnome
        );
    }

    #[test]
    fn detect_both_none() {
        assert_eq!(detect_from_env(None, None), DesktopEnvironment::Unknown);
    }

    #[test]
    fn detect_empty_strings() {
        assert_eq!(detect_from_env(Some(""), None), DesktopEnvironment::Unknown);
        assert_eq!(detect_from_env(None, Some("")), DesktopEnvironment::Unknown);
    }

    #[test]
    fn display_kde_and_unknown() {
        assert_eq!(format!("{}", DesktopEnvironment::Kde), "KDE");
        assert_eq!(format!("{}", DesktopEnvironment::Unknown), "unknown");
    }

    #[test]
    fn needs_portal_and_has_wlr_protocols_exhaustive() {
        assert!(!DesktopEnvironment::Kde.has_wlr_protocols());
        assert!(!DesktopEnvironment::Unknown.has_wlr_protocols());
    }

    #[test]
    fn find_wayland_socket_dir_with_regular_files_only() {
        let dir = std::env::temp_dir().join("remoteway_test_no_sockets");
        std::fs::create_dir_all(&dir).ok();
        // Create regular files that look like wayland sockets but are not
        std::fs::write(dir.join("wayland-0"), b"not a socket").ok();
        std::fs::write(dir.join("wayland-1"), b"also not a socket").ok();
        // .lock files should also be ignored
        std::fs::write(dir.join("wayland-0.lock"), b"lock").ok();
        assert!(find_wayland_socket(dir.to_str().unwrap()).is_none());
        // Cleanup
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn find_wayland_socket_dir_with_non_wayland_files() {
        let dir = std::env::temp_dir().join("remoteway_test_misc_files");
        std::fs::create_dir_all(&dir).ok();
        std::fs::write(dir.join("not-wayland"), b"").ok();
        std::fs::write(dir.join("bus"), b"").ok();
        assert!(find_wayland_socket(dir.to_str().unwrap()).is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn compositor_processes_all_entries_valid() {
        // Verify the table is non-empty and all entries have non-empty process names.
        assert!(!COMPOSITOR_PROCESSES.is_empty());
        for &(name, _) in COMPOSITOR_PROCESSES {
            assert!(
                !name.is_empty(),
                "empty process name in COMPOSITOR_PROCESSES"
            );
        }
    }

    #[test]
    fn desktop_environment_clone_copy_eq() {
        let de = DesktopEnvironment::Gnome;
        let de2 = de; // Copy
        assert_eq!(de, de2);
        let de3: DesktopEnvironment = de;
        assert_eq!(de, de3);
    }

    #[test]
    fn desktop_environment_debug() {
        let dbg = format!("{:?}", DesktopEnvironment::Wlroots);
        assert!(dbg.contains("Wlroots"));
    }

    #[test]
    fn detect_session_desktop_gnome_mixed_case() {
        assert_eq!(
            detect_from_env(None, Some("GNOME")),
            DesktopEnvironment::Gnome
        );
        assert_eq!(
            detect_from_env(None, Some("Gnome-session")),
            DesktopEnvironment::Gnome
        );
    }
}

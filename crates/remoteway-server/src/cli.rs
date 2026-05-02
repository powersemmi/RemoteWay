use clap::Parser;

#[derive(Clone, Debug, clap::ValueEnum)]
pub enum CaptureBackendArg {
    /// Auto-detect best available (ext-image-capture → wlr-screencopy → portal)
    Auto,
    /// wlr-screencopy-unstable-v1 (Hyprland, Sway, wlroots)
    WlrScreencopy,
    /// ext-image-capture-source-v1 (newer standard)
    ExtImageCapture,
    /// xdg-desktop-portal Screencast over `PipeWire` (GNOME, KDE; debug path on wlroots).
    /// Requires the server to be built with `--features gnome`.
    Portal,
}

#[derive(Clone, Debug, clap::ValueEnum)]
pub enum CompressArg {
    Lz4,
    Zstd,
}

#[derive(Parser)]
#[command(
    name = "remoteway-server",
    about = "RemoteWay server: capture → compress → transport + input inject"
)]
pub struct Cli {
    /// Capture backend to use
    #[arg(long, value_enum, default_value = "auto")]
    pub capture: CaptureBackendArg,

    /// Compression algorithm
    #[arg(long, value_enum, default_value = "lz4")]
    pub compress: CompressArg,

    /// Output name to capture (e.g. "DP-1", "HDMI-A-1")
    #[arg(long)]
    pub output: Option<String>,

    /// Capture a specific window by `app_id` (e.g. "org.mozilla.firefox").
    /// Requires ext-image-capture protocol support. Mutually exclusive with --output.
    #[arg(long, conflicts_with = "output")]
    pub app_id: Option<String>,

    /// Command to launch (with `WAYLAND_DISPLAY` pointing to captured compositor)
    #[arg(last = true)]
    pub command: Vec<String>,
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;

    #[test]
    fn defaults() {
        let cli = Cli::parse_from(["remoteway-server"]);
        assert!(matches!(cli.capture, CaptureBackendArg::Auto));
        assert!(matches!(cli.compress, CompressArg::Lz4));
        assert!(cli.output.is_none());
        assert!(cli.app_id.is_none());
        assert!(cli.command.is_empty());
    }

    #[test]
    fn capture_backend_wlr() {
        let cli = Cli::parse_from(["remoteway-server", "--capture", "wlr-screencopy"]);
        assert!(matches!(cli.capture, CaptureBackendArg::WlrScreencopy));
    }

    #[test]
    fn capture_backend_ext() {
        let cli = Cli::parse_from(["remoteway-server", "--capture", "ext-image-capture"]);
        assert!(matches!(cli.capture, CaptureBackendArg::ExtImageCapture));
    }

    #[test]
    fn capture_backend_portal() {
        let cli = Cli::parse_from(["remoteway-server", "--capture", "portal"]);
        assert!(matches!(cli.capture, CaptureBackendArg::Portal));
    }

    #[test]
    fn compress_zstd() {
        let cli = Cli::parse_from(["remoteway-server", "--compress", "zstd"]);
        assert!(matches!(cli.compress, CompressArg::Zstd));
    }

    #[test]
    fn output_option() {
        let cli = Cli::parse_from(["remoteway-server", "--output", "DP-1"]);
        assert_eq!(cli.output.as_deref(), Some("DP-1"));
    }

    #[test]
    fn command_after_double_dash() {
        let cli = Cli::parse_from(["remoteway-server", "--", "firefox", "--headless"]);
        assert_eq!(cli.command, vec!["firefox", "--headless"]);
    }

    #[test]
    fn all_options() {
        let cli = Cli::parse_from([
            "remoteway-server",
            "--capture",
            "wlr-screencopy",
            "--compress",
            "zstd",
            "--output",
            "HDMI-A-1",
            "--",
            "weston-terminal",
        ]);
        assert!(matches!(cli.capture, CaptureBackendArg::WlrScreencopy));
        assert!(matches!(cli.compress, CompressArg::Zstd));
        assert_eq!(cli.output.as_deref(), Some("HDMI-A-1"));
        assert_eq!(cli.command, vec!["weston-terminal"]);
    }

    #[test]
    fn app_id_option() {
        let cli = Cli::parse_from(["remoteway-server", "--app-id", "org.mozilla.firefox"]);
        assert_eq!(cli.app_id.as_deref(), Some("org.mozilla.firefox"));
        assert!(cli.output.is_none());
    }

    #[test]
    fn app_id_with_command() {
        let cli = Cli::parse_from([
            "remoteway-server",
            "--app-id",
            "org.mozilla.firefox",
            "--",
            "firefox",
            "--headless",
        ]);
        assert_eq!(cli.app_id.as_deref(), Some("org.mozilla.firefox"));
        assert_eq!(cli.command, vec!["firefox", "--headless"]);
    }

    #[test]
    fn app_id_conflicts_with_output() {
        let result = Cli::try_parse_from([
            "remoteway-server",
            "--app-id",
            "firefox",
            "--output",
            "DP-1",
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn app_id_with_ext_capture() {
        let cli = Cli::parse_from([
            "remoteway-server",
            "--app-id",
            "firefox",
            "--capture",
            "ext-image-capture",
        ]);
        assert_eq!(cli.app_id.as_deref(), Some("firefox"));
        assert!(matches!(cli.capture, CaptureBackendArg::ExtImageCapture));
    }
}

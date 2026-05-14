use clap::Parser;

#[derive(Clone, Debug, clap::ValueEnum)]
pub enum CaptureBackendArg {
    /// Auto-detect best available (ext-image-capture → wlr-screencopy → portal)
    Auto,
    /// wlr-screencopy-unstable-v1 (Hyprland, Sway, wlroots)
    WlrScreencopy,
    /// ext-image-capture-source-v1 (modern Wayland protocol)
    ExtImageCapture,
    /// xdg-desktop-portal Screencast over `PipeWire` via `GStreamer` (GNOME, KDE).
    /// Requires the server to be built with `--features portal`.
    #[cfg(feature = "portal")]
    Portal,
}

#[derive(Clone, Debug, clap::ValueEnum, PartialEq, Eq)]
pub enum CompressArg {
    /// No compression — frames go on the wire as raw bytes. Use on
    /// loopback / fast LAN where LZ4/zstd CPU cost dominates over
    /// bandwidth savings, or to profile whether compression is the
    /// pipeline bottleneck.
    None,
    /// Fast LZ4 block compression (default).
    Lz4,
    /// Higher-ratio zstd compression.
    Zstd,
}

impl CompressArg {
    /// Project to the runtime [`CompressorKind`] used by the compress
    /// pipeline. Server and client must agree on the same kind via
    /// matching `--compress` flags — there's no in-band negotiation
    /// of the per-region payload format.
    pub fn to_kind(&self) -> remoteway_compress::compressor::CompressorKind {
        match self {
            CompressArg::None => remoteway_compress::compressor::CompressorKind::None,
            CompressArg::Lz4 => remoteway_compress::compressor::CompressorKind::Lz4,
            CompressArg::Zstd => remoteway_compress::compressor::CompressorKind::Zstd,
        }
    }
}

#[derive(Clone, Copy, Debug, clap::ValueEnum, PartialEq, Eq)]
pub enum DownscaleFilterArg {
    /// Area-weighted average over the source rectangle each destination pixel
    /// covers (proper box filter). Preserves anti-aliasing, hides text strokes
    /// less, and feeds a clean image to the client-side upscaler. The right
    /// default for almost all use cases.
    Box,
    /// Nearest-neighbor: pick a single source pixel per destination pixel.
    /// Cheapest CPU cost but causes severe aliasing — only useful on very
    /// constrained hosts where the box filter cannot keep up with the frame
    /// rate.
    Nearest,
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

    /// Capture frame rate limit (10–500 FPS, default 100).
    /// Skips frames arriving faster than the configured rate to prevent
    /// pipeline congestion during rapid screen changes.
    #[arg(long, default_value = "100")]
    pub capture_fps: u32,

    /// Server-side downscale factor before compression (0.1–1.0).
    /// 1.0 = native resolution, 0.5 = half, 0.25 = quarter.
    #[arg(long, default_value = "1.0")]
    pub scale: f64,

    /// Filter used by the server-side downscale (active when --scale < 1.0).
    /// `box` averages the covered source area per destination pixel and is
    /// the right default; `nearest` is faster but produces stair-stepped
    /// edges that the client-side FSR upscale cannot recover.
    #[arg(long, value_enum, default_value = "box")]
    pub downscale_filter: DownscaleFilterArg,

    /// Open portal source-selection dialog, save restore token, then exit.
    /// Requires `--features portal`. Run once from the desktop (not over SSH)
    /// to authorize screen capture so later SSH sessions can skip the dialog.
    #[cfg(feature = "portal")]
    #[arg(long)]
    pub select_source: bool,
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

    #[cfg(feature = "portal")]
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

use clap::Parser;
use remoteway_interpolate::backend::BackendKind;

#[derive(Clone, Debug, clap::ValueEnum)]
pub enum CompressArg {
    Lz4,
    Zstd,
}

/// Mirrors `remoteway_server::cli::CaptureBackendArg`; forwarded to the server
/// via `--capture` over SSH.
#[derive(Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum CaptureBackendArg {
    /// Auto-detect on the server (ext-image-capture → wlr-screencopy → portal).
    Auto,
    /// wlr-screencopy-unstable-v1 (Hyprland, Sway, wlroots).
    WlrScreencopy,
    /// ext-image-capture-source-v1 (modern Wayland protocol).
    ExtImageCapture,
    /// xdg-desktop-portal Screencast over `PipeWire` via `GStreamer`.
    /// Server must be built with `--features portal`.
    Portal,
}

impl CaptureBackendArg {
    /// String value as the server's clap parser expects it.
    fn as_server_arg(&self) -> &'static str {
        match self {
            CaptureBackendArg::Auto => "auto",
            CaptureBackendArg::WlrScreencopy => "wlr-screencopy",
            CaptureBackendArg::ExtImageCapture => "ext-image-capture",
            CaptureBackendArg::Portal => "portal",
        }
    }
}

fn parse_backend_kind(s: &str) -> Result<BackendKind, String> {
    s.parse()
}

#[derive(Parser)]
#[command(
    name = "remoteway",
    about = "RemoteWay client: connect to remote Wayland compositor via SSH"
)]
pub struct Cli {
    /// Remote host in [user@]host format
    pub host: String,

    /// Capture backend for the remote server
    #[arg(long, value_enum, default_value = "auto")]
    pub capture: CaptureBackendArg,

    /// Preferred compression algorithm
    #[arg(long, value_enum, default_value = "lz4")]
    pub compress: CompressArg,

    /// Disable frame interpolation
    #[arg(long, default_value = "false")]
    pub no_interpolate: bool,

    /// Interpolation backend to use (overrides auto-detection).
    /// Auto order: fsr3 → fsr2 → linear-blend.
    /// Available: fsr3, fsr2, fsr2-rife, linear-blend
    #[arg(long, value_parser = parse_backend_kind)]
    pub interpolation_backend: Option<BackendKind>,

    /// Server-side downscale factor forwarded to remoteway-server (0.1–1.0).
    /// 1.0 = native, 0.5 = half resolution before compression.
    #[arg(long, default_value = "1.0")]
    pub server_scale: f64,

    /// Client-side upscale factor applied after receiving (1.0–2.0).
    /// 1.0 = display as-is, 2.0 = double the received frame size.
    #[arg(long, default_value = "1.0")]
    pub upscale: f64,

    /// Capture a specific window by `app_id` on the remote side
    /// (e.g. "org.mozilla.firefox"). Forwarded to remoteway-server as --app-id.
    #[arg(long)]
    pub app_id: Option<String>,

    /// Server-side capture FPS limit (10–500, default 100).
    /// Caps frame capture rate to prevent pipeline congestion.
    #[arg(long, default_value = "100")]
    pub capture_fps: u32,

    /// Path to remoteway-server on remote host
    #[arg(long, default_value = "remoteway-server")]
    pub server_bin: String,

    /// Additional SSH options (e.g. "-p 2222")
    #[arg(long, allow_hyphen_values = true)]
    pub ssh_opt: Vec<String>,

    /// Command to launch on the remote side (passed after --)
    #[arg(last = true)]
    pub command: Vec<String>,
}

impl Cli {
    /// Build the full SSH command line.
    pub fn ssh_command(&self) -> Vec<String> {
        let mut args = vec!["ssh".to_string()];

        for opt in &self.ssh_opt {
            args.push(opt.clone());
        }

        args.push(self.host.clone());
        args.push(self.server_bin.clone());

        if !matches!(self.capture, CaptureBackendArg::Auto) {
            args.push("--capture".to_string());
            args.push(self.capture.as_server_arg().to_string());
        }

        if let Some(ref app_id) = self.app_id {
            args.push("--app-id".to_string());
            args.push(app_id.clone());
        }

        // Forward capture FPS limit.
        args.push("--capture-fps".to_string());
        args.push(self.capture_fps.to_string());

        // Forward server-side scale factor.
        if (self.server_scale - 1.0).abs() > f64::EPSILON {
            args.push("--scale".to_string());
            args.push(self.server_scale.to_string());
        }

        if !self.command.is_empty() {
            args.push("--".to_string());
            args.extend(self.command.iter().cloned());
        }

        args
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;

    #[test]
    fn ssh_command_basic() {
        let cli = Cli::parse_from(["remoteway", "user@host"]);
        let cmd = cli.ssh_command();
        assert_eq!(cmd, vec!["ssh", "user@host", "remoteway-server"]);
    }

    #[test]
    fn ssh_command_with_app() {
        let cli = Cli::parse_from(["remoteway", "host", "--", "firefox"]);
        let cmd = cli.ssh_command();
        assert_eq!(
            cmd,
            vec!["ssh", "host", "remoteway-server", "--", "firefox"]
        );
    }

    #[test]
    fn ssh_command_with_opts() {
        let cli = Cli::parse_from([
            "remoteway",
            "host",
            "--ssh-opt",
            "-p 2222",
            "--server-bin",
            "/usr/bin/remoteway-server",
        ]);
        let cmd = cli.ssh_command();
        assert_eq!(
            cmd,
            vec!["ssh", "-p 2222", "host", "/usr/bin/remoteway-server"]
        );
    }

    #[test]
    fn ssh_command_with_multiple_opts() {
        let cli = Cli::parse_from([
            "remoteway",
            "host",
            "--ssh-opt",
            "-o StrictHostKeyChecking=no",
            "--ssh-opt",
            "-p 2222",
        ]);
        let cmd = cli.ssh_command();
        assert_eq!(cmd[1], "-o StrictHostKeyChecking=no");
        assert_eq!(cmd[2], "-p 2222");
    }

    #[test]
    fn defaults() {
        let cli = Cli::parse_from(["remoteway", "host"]);
        assert_eq!(cli.host, "host");
        assert!(!cli.no_interpolate);
        assert_eq!(cli.server_bin, "remoteway-server");
        assert!(cli.ssh_opt.is_empty());
        assert!(cli.command.is_empty());
        assert!(matches!(cli.compress, CompressArg::Lz4));
    }

    #[test]
    fn no_interpolate_flag() {
        let cli = Cli::parse_from(["remoteway", "host", "--no-interpolate"]);
        assert!(cli.no_interpolate);
    }

    #[test]
    fn compress_zstd() {
        let cli = Cli::parse_from(["remoteway", "host", "--compress", "zstd"]);
        assert!(matches!(cli.compress, CompressArg::Zstd));
    }

    #[test]
    fn server_scale_default() {
        let cli = Cli::parse_from(["remoteway", "host"]);
        assert!((cli.server_scale - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn server_scale_custom() {
        let cli = Cli::parse_from(["remoteway", "host", "--server-scale", "0.5"]);
        assert!((cli.server_scale - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn upscale_default() {
        let cli = Cli::parse_from(["remoteway", "host"]);
        assert!((cli.upscale - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn upscale_custom() {
        let cli = Cli::parse_from(["remoteway", "host", "--upscale", "1.5"]);
        assert!((cli.upscale - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn ssh_command_forwards_scale() {
        let cli = Cli::parse_from(["remoteway", "host", "--server-scale", "0.5"]);
        let cmd = cli.ssh_command();
        let scale_pos = cmd.iter().position(|s| s == "--scale").unwrap();
        assert_eq!(cmd[scale_pos + 1], "0.5");
    }

    #[test]
    fn ssh_command_no_scale_when_default() {
        let cli = Cli::parse_from(["remoteway", "host"]);
        let cmd = cli.ssh_command();
        assert!(!cmd.contains(&"--scale".to_string()));
    }

    #[test]
    fn resolution_zero_rejected() {
        let result = Cli::try_parse_from(["remoteway", "host", "--resolution", "0x1080"]);
        assert!(result.is_err());
    }

    #[test]
    fn interpolation_backend_flag() {
        let cli = Cli::parse_from([
            "remoteway",
            "host",
            "--interpolation-backend",
            "linear-blend",
        ]);
        assert_eq!(cli.interpolation_backend.unwrap(), BackendKind::LinearBlend);
    }

    #[test]
    fn interpolation_backend_fsr3_alias() {
        let cli = Cli::parse_from(["remoteway", "host", "--interpolation-backend", "fsr3"]);
        assert_eq!(cli.interpolation_backend.unwrap(), BackendKind::Fsr3);
    }

    #[test]
    fn interpolation_backend_invalid() {
        let result = Cli::try_parse_from([
            "remoteway",
            "host",
            "--interpolation-backend",
            "nonexistent",
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn interpolation_backend_default_none() {
        let cli = Cli::parse_from(["remoteway", "host"]);
        assert!(cli.interpolation_backend.is_none());
    }

    #[test]
    fn app_id_forwarded_in_ssh_command() {
        let cli = Cli::parse_from([
            "remoteway",
            "host",
            "--app-id",
            "org.mozilla.firefox",
            "--",
            "firefox",
        ]);
        let cmd = cli.ssh_command();
        assert!(cmd.contains(&"--app-id".to_string()));
        assert!(cmd.contains(&"org.mozilla.firefox".to_string()));
        // --app-id should come before --
        let app_id_pos = cmd.iter().position(|s| s == "--app-id").unwrap();
        let dash_pos = cmd.iter().position(|s| s == "--").unwrap();
        assert!(app_id_pos < dash_pos);
    }

    #[test]
    fn app_id_default_none() {
        let cli = Cli::parse_from(["remoteway", "host"]);
        assert!(cli.app_id.is_none());
    }

    #[test]
    fn capture_default_auto() {
        let cli = Cli::parse_from(["remoteway", "host"]);
        assert!(matches!(cli.capture, CaptureBackendArg::Auto));
    }

    #[test]
    fn capture_portal() {
        let cli = Cli::parse_from(["remoteway", "host", "--capture", "portal"]);
        assert!(matches!(cli.capture, CaptureBackendArg::Portal));
    }

    #[test]
    fn capture_auto_not_forwarded() {
        let cli = Cli::parse_from(["remoteway", "host"]);
        let cmd = cli.ssh_command();
        assert!(!cmd.contains(&"--capture".to_string()));
    }

    #[test]
    fn capture_forwarded_in_ssh_command() {
        let cli = Cli::parse_from(["remoteway", "host", "--capture", "wlr-screencopy"]);
        let cmd = cli.ssh_command();
        let cap_pos = cmd.iter().position(|s| s == "--capture").unwrap();
        assert_eq!(cmd[cap_pos + 1], "wlr-screencopy");
    }

    #[test]
    fn capture_before_double_dash() {
        let cli = Cli::parse_from([
            "remoteway",
            "host",
            "--capture",
            "ext-image-capture",
            "--",
            "firefox",
        ]);
        let cmd = cli.ssh_command();
        let cap_pos = cmd.iter().position(|s| s == "--capture").unwrap();
        let dash_pos = cmd.iter().position(|s| s == "--").unwrap();
        assert!(cap_pos < dash_pos);
    }
}

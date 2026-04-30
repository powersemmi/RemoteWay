# RemoteWay

Low-latency Wayland remote desktop over SSH. Forward a remote Wayland compositor's display and input to your local machine with zero configuration &mdash; just `remoteway user@host`.

## How it works

```
Local machine                              Remote machine
 (client)                 SSH                (server)
+-----------+     stdin/stdout pipe     +-----------+
| Display   | <---- decompress <------- | Capture   |
| (Wayland) |       + interpolate       | (Wayland) |
|           |                           |           |
| Input     | -----> serialize --------> | Inject    |
| (pointer, |       (priority queue)    | (virtual  |
|  keyboard)|                           |  pointer) |
+-----------+                           +-----------+
```

RemoteWay captures frames from the remote compositor, delta-encodes and compresses them, streams them over SSH, decompresses on the client, optionally interpolates between frames, and displays them in a local Wayland window. Input events travel the reverse path with the highest priority.

## Features

- **Zero config**: uses your existing SSH setup, no VPN or port forwarding needed
- **Low latency**: thread-per-core pipeline, lock-free SPSC queues, SIMD delta encoding, `SCHED_FIFO` real-time priorities
- **Wayland native**: wlr-screencopy, ext-image-capture, xdg-desktop-portal + PipeWire (GNOME)
- **Adaptive compression**: LZ4 (fast) or Zstd (better ratio), automatic delta encoding
- **Frame interpolation**: GPU-accelerated backends (wgpu, AMD FSR2/3, NVIDIA Optical Flow) with CPU fallback
- **Multi-window**: each remote toplevel gets its own local xdg_toplevel surface
- **Input forwarding**: pointer, keyboard, scroll &mdash; with local cursor overlay for instant feedback
- **Clipboard forwarding**: text, HTML, PNG between local and remote

## Requirements

### Build

- Rust 1.85+ (edition 2024)
- System libraries:
  - `libwayland-dev` (client + server)
  - `libwayland-protocols` (wlr + misc)

Optional (for GNOME support, `--features gnome`):
  - `libpipewire-0.3-dev`
  - `libei-dev`

### Runtime

- **Remote machine**: Wayland compositor (Sway, Hyprland, Niri, GNOME, KDE)
- **Local machine**: Wayland compositor
- SSH access to the remote machine
- `remoteway-server` installed on the remote machine

## Installation

```bash
# Clone and build
git clone https://github.com/user/remoteway.git
cd remoteway
cargo build --release

# Install binaries
sudo install -m 755 target/release/remoteway-client /usr/local/bin/remoteway
sudo install -m 755 target/release/remoteway-server /usr/local/bin/remoteway-server
```

The server binary must be in `$PATH` on the remote machine (or specify its path with `--server-bin`).

### With GNOME support

```bash
cargo build --release --features gnome
```

This enables xdg-desktop-portal screen capture and libei input injection for GNOME/Mutter.

## Usage

### Basic

```bash
# Connect to a remote machine and see its desktop
remoteway user@host

# Run a specific application
remoteway user@host -- firefox

# Run with verbose logging
RUST_LOG=remoteway=debug remoteway user@host
```

### Options

```
remoteway [OPTIONS] <HOST> [-- <COMMAND>...]

Arguments:
  <HOST>         Remote host in [user@]host format
  [COMMAND]...   Command to launch on the remote side

Options:
  --capture <BACKEND>          Capture backend for the remote server
                               [auto|wlr-screencopy|ext-image-capture|portal]
                               [default: auto]
  --compress <lz4|zstd>        Compression algorithm [default: lz4]
  --no-interpolate             Disable frame interpolation
  --interpolation-backend <BACKEND>
                               Interpolation backend (overrides auto-detection)
                               [linear-blend|wgpu-optical-flow|fsr2|fsr3|nvidia-optical-flow|rife]
  --resolution <WxH>           Target resolution for server-side downscaling
                               (e.g. 1920x1080). Omit for native resolution
  --app-id <APP_ID>            Capture a specific window by app_id on the
                               remote side (e.g. "org.mozilla.firefox")
  --server-bin <PATH>          Path to remoteway-server on remote host
                               [default: remoteway-server]
  --ssh-opt <OPT>              Additional SSH options (repeatable)
                               e.g. --ssh-opt "-p 2222"
```

### Examples

```bash
# Custom SSH port
remoteway --ssh-opt "-p 2222" user@host

# Disable interpolation for lowest latency
remoteway --no-interpolate user@host

# Use Zstd compression (better ratio, slightly more CPU)
remoteway --compress zstd user@host

# Force portal capture on the remote server
remoteway --capture portal user@host

# Downscale remote 4K display to 1080p
remoteway --resolution 1920x1080 user@host

# Capture a specific remote window
remoteway --app-id org.mozilla.firefox user@host -- firefox

# Specify server binary location
remoteway --server-bin /opt/remoteway/remoteway-server user@host

# Run a remote terminal
remoteway user@host -- weston-terminal
```

### Server options

The server is normally launched automatically by the client via SSH. For manual use:

```
remoteway-server [OPTIONS] [-- <COMMAND>...]

Options:
  --capture <BACKEND>          Capture backend [default: auto]
                               [auto|wlr-screencopy|ext-image-capture|portal]
  --compress <lz4|zstd>        Compression algorithm [default: lz4]
  --output <NAME>              Output to capture (e.g. "DP-1")
  --app-id <APP_ID>            Capture a specific window by app_id
                               (e.g. "org.mozilla.firefox").
                               Mutually exclusive with --output
```

## Architecture

```
[SERVER]
wlr-screencopy (Core 1, SCHED_FIFO 90)
    | rtrb SPSC
Delta encode + LZ4 compress (Core 2)
    | rtrb SPSC
SSH stdout (priority queue: input > anchor > frame)

Input receive <- SSH stdin (Core 0, SCHED_FIFO 99)
    | immediately
wlr-virtual-pointer / virtual-keyboard inject

[CLIENT]
SSH stdout -> StreamParser (priority queue)
    | rtrb SPSC
Decompress + delta reconstruct (Core 1-2)
    | optional: FrameInterpolator (GPU)
wl_shm -> wl_surface.commit (Core 3)

Input capture -> SSH stdin (SCHED_FIFO 99, bypasses frame queue)
Cursor overlay - drawn locally, instant
```

### Crate structure

| Crate | Purpose |
|-------|---------|
| `remoteway-proto` | Wire protocol types (zerocopy, fixed-size headers) |
| `remoteway-core` | BufferPool, ThreadConfig, LatencyHistogram, BandwidthMeter |
| `remoteway-capture` | Screen capture (wlr-screencopy, ext-image-capture, PipeWire) |
| `remoteway-compress` | Delta encoding (SIMD), LZ4/Zstd compression |
| `remoteway-transport` | SSH multiplexed transport with priority queues |
| `remoteway-input` | Input capture (client) and injection (server, wlr + libei) |
| `remoteway-display` | Wayland display, surface management, cursor overlay |
| `remoteway-interpolate` | Frame interpolation (CPU blend, wgpu, FSR2/3, NVIDIA OF) |
| `remoteway-server` | Server binary |
| `remoteway-client` | Client binary |

## Performance tuning

### Real-time scheduling

For lowest latency, grant the binaries real-time scheduling capability:

```bash
# Allow real-time scheduling without root
sudo setcap cap_sys_nice=ep /usr/local/bin/remoteway
sudo setcap cap_sys_nice=ep /usr/local/bin/remoteway-server
```

Or set the RLIMIT:

```bash
# /etc/security/limits.d/remoteway.conf
your_user  soft  rtprio  99
your_user  hard  rtprio  99
your_user  soft  memlock unlimited
your_user  hard  memlock unlimited
```

### Memory locking

Both binaries call `mlockall(MCL_CURRENT | MCL_FUTURE)` at startup to prevent page faults on the hot path. If you see a warning about `mlockall`, increase `RLIMIT_MEMLOCK`:

```bash
ulimit -l unlimited
```

### Profiling

RemoteWay integrates [Tracy](https://github.com/wolfpld/tracy) for frame-level profiling:

```bash
# Build with Tracy support
cargo build --release --features tracy

# Run with Tracy profiler attached
remoteway-server --features tracy
```

CPU flamegraphs:

```bash
cargo install flamegraph
cargo flamegraph --bin remoteway-server -- --capture auto
```

### Benchmarks

```bash
# Compression pipeline benchmarks (delta encode, LZ4, full pipeline)
cargo bench -p remoteway-compress

# Transport throughput
cargo bench -p remoteway-transport

# Buffer pool acquire/release
cargo bench -p remoteway-core
```

## Compositor support

| Compositor | Capture | Input | Status |
|-----------|---------|-------|--------|
| Sway | wlr-screencopy | wlr-virtual-pointer | Fully supported |
| Hyprland | wlr-screencopy / ext-image-capture | wlr-virtual-pointer | Fully supported |
| GNOME/Mutter | xdg-desktop-portal + PipeWire | libei | Requires `--features gnome` |
| KDE Plasma | xdg-desktop-portal + PipeWire | libei | Requires `--features gnome` |
| wlroots-based | Auto-detected | Auto-detected | Fully supported |

## Development

```bash
# Format, lint, test (run before every commit)
cargo fmt --all
cargo clippy --all -- -D warnings
cargo test --workspace

# Coverage report
cargo llvm-cov --workspace --html
open target/llvm-cov/html/index.html

# Fuzz testing
cargo install cargo-fuzz
cd fuzz
cargo fuzz run fuzz_frame_header
cargo fuzz run fuzz_stream_parser
```

## License

Licensed under the [MIT License](LICENSE).

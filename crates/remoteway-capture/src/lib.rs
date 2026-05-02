//! Screen capture module for [RemoteWay].
//!
//! This crate provides the core screen capture infrastructure used by the `RemoteWay` remote
//! desktop application on Wayland. It abstracts over multiple capture backends, output
//! enumeration, desktop environment detection, and efficient frame transport so that
//! higher-level layers can request screen frames without worrying about the underlying
//! protocol details.
//!
//! # Architecture
//!
//! The capture pipeline is organised around the [`backend::CaptureBackend`] trait. Each
//! concrete backend implements this trait to produce [`backend::CapturedFrame`]s in one of
//! several supported [`backend::PixelFormat`]s. Client code normally obtains a backend
//! through the auto-detection logic in [`detect`], which probes the available Wayland
//! protocols and returns the best backend for the current compositor.
//!
//! Captured frames are transported from the capture thread to consumers via a
//! single-producer-single-consumer ring buffer defined in [`thread`]. The thread module
//! also owns the capture loop that drives the selected backend.
//!
//! # Module overview
//!
//! | Module | Purpose |
//! |---|---|
//! | [`backend`] | [`CaptureBackend`](backend::CaptureBackend) trait, [`CapturedFrame`](backend::CapturedFrame), and [`PixelFormat`](backend::PixelFormat). |
//! | [`desktop_detect`] | Heuristics for identifying the active desktop environment (KDE, GNOME, wlroots, …). |
//! | [`detect`] | Auto-detection of the best available capture backend for the current compositor. |
//! | [`error`] | [`CaptureError`](error::CaptureError) error type and related helpers. |
//! | [`ext_capture`] | Backend using the `ext-image-capture-source-v1` Wayland protocol. |
//! | [`output`] | Wayland output enumeration (identifying displays / monitors). |
//! | [`portal`] | GStreamer + xdg-desktop-portal backend for GNOME/KDE. |
//! | [`protocols`] | Auto-generated Wayland protocol bindings (internal; not public API). |
//! | [`screencopy`] | Backend using the legacy `wlr-screencopy` protocol. |
//! | [`shm`] | Double-buffered shared-memory (SHM) pool for zero-copy frame storage. |
//! | [`thread`] | Capture thread and SPSC ring-buffer transport. |
//! | [`toplevel`] | Foreign toplevel tracking for per-window capture. |
//!
//! [RemoteWay]: https://github.com/RemoteWay/RemoteWay

pub mod backend;
pub mod desktop_detect;
pub mod detect;
pub mod error;
pub mod ext_capture;
pub mod output;
#[cfg(feature = "portal")]
pub mod portal;
mod protocols;
pub mod screencopy;
pub mod shm;
pub mod thread;
pub mod toplevel;

//! Per-codec encoder backends. Each lives behind its own feature flag.

#[cfg(feature = "h264")]
pub mod h264;

#[cfg(feature = "h265")]
pub mod h265;

#[cfg(feature = "av1")]
pub mod av1;

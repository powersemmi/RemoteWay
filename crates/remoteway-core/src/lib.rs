//! Core abstractions for the **`RemoteWay`** low‑latency frame‑processing
//! pipeline.
//!
//! # Overview
//!
//! This crate provides the foundational building blocks on which the rest of
//! the `RemoteWay` stack is constructed. Every type is designed for the hot path:
//! lock‑free where possible, cache‑line aligned, and compatible with
//! core‑pinning and real‑time scheduling.
//!
//! # Modules
//!
//! | Module | Role |
//! |---|---|
//! | [`bandwidth`] | Sliding‑window bandwidth metering |
//! | [`buffer_pool`] | Lock‑free pre‑allocated frame buffer pool |
//! | [`frame_handle`] | Lightweight handles for zero‑copy frame passing |
//! | [`latency`] | End‑to‑end latency histogram |
//! | [`pipeline`] | Common [`PipelineStage`](pipeline::PipelineStage) trait |
//! | [`thread_config`] | Core‑pinning & real‑time scheduling configuration |
//!
//! # Usage
//!
//! A typical pipeline thread acquires a [`FrameHandle`] from the
//! [`BufferPool`](buffer_pool::BufferPool), processes the frame through a
//! chain of [`PipelineStage`](pipeline::PipelineStage) implementations, tracks
//! the latency in a [`LatencyHistogram`](latency::LatencyHistogram), reports
//! throughput via a [`BandwidthMeter`](bandwidth::BandwidthMeter), and was
//! spawned with [`ThreadConfig`](thread_config::ThreadConfig) to ensure
//! deterministic scheduling.
//!
//! ```
//! use remoteway_core::{
//!     bandwidth::BandwidthMeter,
//!     buffer_pool::BufferPool,
//!     latency::LatencyHistogram,
//!     pipeline::PipelineStage,
//!     thread_config::ThreadConfig,
//! };
//!
//! let pool = BufferPool::new(8, 4096);
//! pool.acquire(0).unwrap();
//! ```

/// Sliding-window bandwidth meter.
///
/// See [`bandwidth::BandwidthMeter`] for details.
pub mod bandwidth;

/// Lock-free pre-allocated frame buffer pool.
///
/// See [`buffer_pool::BufferPool`] for details.
pub mod buffer_pool;

/// Lightweight handles for passing frame data between pipeline stages.
///
/// See [`frame_handle::FrameHandle`] for details.
pub mod frame_handle;

/// Histogram for measuring end-to-end latency.
///
/// See [`latency::LatencyHistogram`] for details.
pub mod latency;

/// Common `PipelineStage` trait.
///
/// See [`pipeline::PipelineStage`] for details.
pub mod pipeline;

/// Core-pinning and real-time scheduling configuration.
///
/// See [`thread_config::ThreadConfig`] for details.
pub mod thread_config;

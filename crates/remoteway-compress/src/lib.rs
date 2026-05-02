//! Compression pipeline for the `RemoteWay` remote file synchronization engine.
//!
//! This crate provides a multi-stage compression pipeline combining
//! [delta encoding](delta) with [LZ4](lz4)/[zstd](compressor) block compression,
//! plus [SIMD-accelerated](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data)
//! operations for high-throughput data reduction.
//!
//! # Architecture
//!
//! The pipeline flows through these stages:
//!
//! 1. **[`delta`]** — Computes and applies delta (diff) encodings between
//!    successive versions of file data, so only changed bytes are transmitted.
//! 2. **[`lz4`]** — Fast LZ4 block compression for low-latency paths.
//! 3. **[`compressor`]** — Zstandard (zstd) compression for high-ratio scenarios.
//! 4. **[`pipeline`]** — Orchestrates the above stages into a unified encode /
//!    decode pipeline.
//! 5. **[`stats`]** — Collects and reports per-stage compression metrics
//!    (ratios, throughput, CPU usage).
//!
//! # Example
//!
//! ```no_run
//! use remoteway_compress::pipeline;
//! // Pipeline usage goes through the `pipeline` module.
//! ```

pub mod compressor;
pub mod delta;
pub mod lz4;
pub mod pipeline;
pub mod stats;

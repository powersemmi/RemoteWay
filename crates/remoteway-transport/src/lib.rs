//! SSH-multiplexed transport layer for the `RemoteWay` remote file
//! synchronization engine.
//!
//! This crate implements a multiplexed protocol over an SSH channel's
//! `stdin`/`stdout` streams, allowing multiple concurrent chunk streams to
//! share a single SSH connection.  Flow control and back-pressure are built in
//! so that a slow consumer cannot stall the entire connection.
//!
//! # Architecture
//!
//! | Module | Role |
//! |---|---|
//! | **[`chunk_sender`]** | Splits payloads into fixed-size chunks and queues them for transmission. |
//! | **[`flow_control`]** | Manages per-stream send windows and applies back-pressure when buffers fill. |
//! | **[`multiplexer`]** | Multiplexes many logical streams onto one SSH channel via tagged frames. |
//! | **[`ssh_transport`]** | Wraps `ssh2` / OpenSSH channels, providing `AsyncRead` + `AsyncWrite` on `stdin`/`stdout`. |
//!
//! # Example
//!
//! ```no_run
//! use remoteway_transport::ssh_transport;
//! // Establish transport via the `ssh_transport` module.
//! ```

pub mod chunk_sender;
pub mod flow_control;
pub mod multiplexer;
pub mod ssh_transport;

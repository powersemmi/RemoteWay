//! Wire protocol types for the **`RemoteWay`** remote desktop system.
//!
//! This crate defines all the low-level message structures, enumerations, and
//! serialization primitives used to communicate between a `RemoteWay` client and
//! server. Every frame sent over the transport begins with a universal
//! [`header`], then carries a type-specific payload determined by the message
//! kind.
//!
//! # Modules
//!
//! | Module | Purpose |
//! |---|---|
//! | [`clipboard`] | Clipboard transfer protocol — copy/paste between hosts. |
//! | [`cursor`] | Cursor position updates and cursor-image metadata. |
//! | [`frame`] | Frame metadata, damage regions, and wire-area descriptors. |
//! | [`handshake`] | Connection handshake with capability negotiation. |
//! | [`header`] | Universal frame header, message-type enum, and framing helpers. |
//! | [`input`] | Input event protocol covering pointer, keyboard, and scroll. |
//! | [`monitor`] | Monitor descriptors, EDID information, and fractional-scale hints. |
//! | [`resize`] | Surface / buffer resize notification events. |
//! | [`target_resolution`] | Client-requested downscaling for bandwidth-adaptive streaming. |
//!
//! # Usage
//!
//! Most consumers will re-export everything they need through a single `use`
//! statement (each module exposes its public types at the module root).  For
//! example:
//!
//! ```ignore
//! use remoteway_proto::header::{FrameHeader, MessageType};
//! use remoteway_proto::handshake::{ClientHello, ServerHello};
//! ```

pub mod clipboard;
pub mod cursor;
pub mod error;
pub mod frame;
pub mod handshake;
pub mod header;
pub mod input;
pub mod monitor;
pub mod resize;
pub mod target_resolution;

pub use error::ProtoError;

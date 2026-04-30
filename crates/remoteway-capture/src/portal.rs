//! xdg-desktop-portal Screencast integration via D-Bus (zbus).
//!
//! Portal methods use the Request/Response pattern: each method call returns
//! a Request object path, and the actual result arrives as a `Response` signal
//! on that path. This module handles the full async handshake correctly.
//!
//! Requires the `gnome` feature flag.

use std::collections::HashMap;
use std::path::PathBuf;

use futures_util::StreamExt;
use zbus::Connection;
use zbus::zvariant::{OwnedObjectPath, OwnedValue, Value};

use crate::error::CaptureError;

/// Path to the file where we persist the portal session identifier.
/// The identifier allows restoring a previously authorized session
/// without showing the portal dialog on subsequent runs.
const PORTAL_ID_FILE: &str = "remoteway/portal-id";

/// Return the path to the portal identifier file.
fn portal_id_path() -> PathBuf {
    // Try XDG_CONFIG_HOME first, fall back to ~/.config
    if let Ok(dir) = std::env::var("XDG_CONFIG_HOME") {
        PathBuf::from(dir).join(PORTAL_ID_FILE)
    } else if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home).join(".config").join(PORTAL_ID_FILE)
    } else {
        PathBuf::from(PORTAL_ID_FILE)
    }
}

/// Load a previously saved portal session identifier.
fn load_portal_id() -> Option<String> {
    let path = portal_id_path();
    std::fs::read_to_string(&path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Save a portal session identifier for future use.
fn save_portal_id(identifier: &str) {
    let path = portal_id_path();
    let path_str = path.to_string_lossy();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Err(e) = std::fs::write(&path, identifier) {
        tracing::warn!(err = %e, path = %path_str, "failed to save portal identifier");
    } else {
        tracing::info!(%identifier, "portal identifier saved for skip-dialog on next run");
    }
}

/// Result of a successful portal screencast session.
pub struct PortalSession {
    /// PipeWire file descriptor for connecting to the PipeWire daemon.
    pub pw_fd: std::os::fd::OwnedFd,
    /// PipeWire node ID of the screencast stream.
    pub pw_node_id: u32,
    /// Stream size reported by the portal (may be used as fallback).
    pub stream_width: u32,
    pub stream_height: u32,
    /// Session handle (D-Bus object path) for cleanup.
    pub session_handle: OwnedObjectPath,
    /// Portal session identifier for restoring this session without a dialog
    /// on subsequent runs. Saved to disk for reuse across invocations.
    pub portal_id: Option<String>,
}

/// Whether to capture a monitor, a specific window, or show both options.
#[derive(Debug, Clone, Copy)]
pub enum PortalSourceType {
    Monitor,
    Window,
    /// Show both monitors and windows in the portal picker (types=3).
    Both,
}

/// Create a screencast session via xdg-desktop-portal.
///
/// This is an async function that must be called from a tokio context.
pub async fn create_screencast_session(
    source_type: PortalSourceType,
    _embed_cursor: bool,
) -> Result<PortalSession, CaptureError> {
    let connection = Connection::session()
        .await
        .map_err(|e| CaptureError::Protocol(format!("D-Bus session connection failed: {e}")))?;

    // 1. CreateSession
    let mut create_opts: HashMap<&str, Value<'_>> = HashMap::new();
    create_opts.insert("handle_token", Value::from("remoteway_create"));
    create_opts.insert("session_handle_token", Value::from("remoteway_session"));

    let create_results = portal_request(
        &connection,
        "CreateSession",
        &(create_opts,),
        "remoteway_create",
    )
    .await?;

    // Extract session_handle from the Response results.
    let session_handle = extract_session_handle(&connection, &create_results)?;
    tracing::debug!(session = %session_handle, "portal session created");

    // 2. SelectSources
    let source_types: u32 = match source_type {
        PortalSourceType::Monitor => 1,
        PortalSourceType::Window => 2,
        PortalSourceType::Both => 3,
    };

    let mut select_opts: HashMap<&str, Value<'_>> = HashMap::new();
    select_opts.insert("handle_token", Value::from("remoteway_select"));
    select_opts.insert("types", Value::U32(source_types));
    select_opts.insert("multiple", Value::Bool(false));
    // Persist mode = 1: remember the selection so the dialog is skipped
    // on subsequent runs after the user authorizes once.
    select_opts.insert("persist_mode", Value::U32(1));

    portal_request(
        &connection,
        "SelectSources",
        &(&session_handle, select_opts),
        "remoteway_select",
    )
    .await?;

    // 3. Start — may trigger a user dialog (e.g. screen/window picker).
    // If we have a previously saved portal token, pass it to restore
    // the session without showing the dialog again.
    // GNOME uses "identifier", KDE uses "restore_token".
    let saved_id = load_portal_id();
    let mut start_opts: HashMap<&str, Value<'_>> = HashMap::new();
    start_opts.insert("handle_token", Value::from("remoteway_start"));
    if let Some(ref id) = saved_id {
        // Pass as both identifier and restore_token — the portal backend
        // will use whichever it understands.
        start_opts.insert("identifier", Value::from(id.as_str()));
        start_opts.insert("restore_token", Value::from(id.as_str()));
        tracing::info!(%id, "using saved portal token to skip dialog");
    }

    let start_results = portal_request(
        &connection,
        "Start",
        &(&session_handle, "", start_opts),
        "remoteway_start",
    )
    .await?;

    // Log all keys from Start response for debugging.
    let start_keys: Vec<&str> = start_results.keys().map(|s| s.as_str()).collect();
    tracing::info!(?start_keys, "portal Start response keys");

    let pw_node_id = extract_pw_node_id(&start_results)?;
    let (stream_width, stream_height) = extract_stream_size(&start_results).unwrap_or((1920, 1080));
    // Extract the portal session identifier/restore_token so we can skip
    // the dialog next time. GNOME returns "identifier", KDE returns "restore_token".
    let portal_id = start_results
        .get("identifier")
        .or_else(|| start_results.get("restore_token"))
        .and_then(|v| <String>::try_from(v.clone()).ok().filter(|s| !s.is_empty()));
    if let Some(ref id) = portal_id {
        tracing::info!(%id, "portal session token received, saving for reuse");
        save_portal_id(id);
    } else {
        tracing::info!("no portal identifier/restore_token — dialog will appear on next run too");
    }
    tracing::info!(
        pw_node_id,
        stream_width,
        stream_height,
        "portal screencast PipeWire node"
    );

    // 4. OpenPipeWireRemote — returns a file descriptor directly (not Request/Response).
    let empty_opts: HashMap<&str, Value<'_>> = HashMap::new();
    let fd_reply = connection
        .call_method(
            Some("org.freedesktop.portal.Desktop"),
            "/org/freedesktop/portal/desktop",
            Some("org.freedesktop.portal.ScreenCast"),
            "OpenPipeWireRemote",
            &(&session_handle, empty_opts),
        )
        .await
        .map_err(|e| CaptureError::Protocol(format!("OpenPipeWireRemote failed: {e}")))?;

    let zbus_fd: zbus::zvariant::OwnedFd = fd_reply
        .body()
        .deserialize()
        .map_err(|e| CaptureError::Protocol(format!("fd deserialize failed: {e}")))?;

    let pw_fd: std::os::fd::OwnedFd = zbus_fd.into();

    Ok(PortalSession {
        pw_fd,
        pw_node_id,
        stream_width,
        stream_height,
        session_handle,
        portal_id,
    })
}

fn get_sender_name(connection: &Connection) -> String {
    connection
        .unique_name()
        .map(|n| n.as_str().trim_start_matches(':').replace('.', "_"))
        .unwrap_or_else(|| "unknown".to_string())
}

/// Call a portal method and wait for the async Response signal.
///
/// Portal methods follow the Request/Response pattern:
/// 1. Method call returns a Request object path immediately
/// 2. The actual result arrives as a `Response` signal on that path
///
/// We subscribe to the message stream *before* making the call to avoid
/// race conditions where the signal arrives before we start listening.
async fn portal_request<B>(
    connection: &Connection,
    method: &str,
    body: &B,
    handle_token: &str,
) -> Result<HashMap<String, OwnedValue>, CaptureError>
where
    B: zbus::zvariant::DynamicType + serde::Serialize,
{
    let sender = get_sender_name(connection);
    let request_path = format!("/org/freedesktop/portal/desktop/request/{sender}/{handle_token}");

    // Subscribe to D-Bus messages BEFORE making the call to avoid missing the signal.
    let mut stream = zbus::MessageStream::from(connection);

    // Make the portal method call.
    connection
        .call_method(
            Some("org.freedesktop.portal.Desktop"),
            "/org/freedesktop/portal/desktop",
            Some("org.freedesktop.portal.ScreenCast"),
            method,
            body,
        )
        .await
        .map_err(|e| CaptureError::Protocol(format!("{method} call failed: {e}")))?;

    // Wait for the Response signal with a timeout.
    let timeout_duration = tokio::time::Duration::from_secs(30);

    let result = tokio::time::timeout(timeout_duration, async {
        while let Some(msg) = stream.next().await {
            let msg = msg.map_err(|e| {
                CaptureError::Protocol(format!("{method}: D-Bus stream error: {e}"))
            })?;
            let hdr = msg.header();

            // Filter for Response signal at the expected Request path.
            let is_response_signal = hdr.message_type() == zbus::message::Type::Signal
                && hdr.member().is_some_and(|m| m.as_str() == "Response")
                && hdr.path().is_some_and(|p| p.as_str() == request_path);
            if is_response_signal {
                let (code, results): (u32, HashMap<String, OwnedValue>) =
                    msg.body().deserialize().map_err(|e| {
                        CaptureError::Protocol(format!("{method} response parse failed: {e}"))
                    })?;

                if code != 0 {
                    return Err(CaptureError::Protocol(format!(
                        "{method} rejected by user or failed (code {code})"
                    )));
                }

                return Ok(results);
            }
        }
        Err(CaptureError::Protocol(format!(
            "{method}: D-Bus connection closed before response"
        )))
    })
    .await;

    match result {
        Ok(inner) => inner,
        Err(_) => Err(CaptureError::Protocol(format!(
            "{method}: timed out waiting for portal response (30s)"
        ))),
    }
}

/// Extract `session_handle` from the CreateSession Response results.
fn extract_session_handle(
    connection: &Connection,
    results: &HashMap<String, OwnedValue>,
) -> Result<OwnedObjectPath, CaptureError> {
    if let Some(v) = results.get("session_handle") {
        // Try proper OwnedValue → OwnedObjectPath conversion.
        if let Ok(path) = <OwnedObjectPath>::try_from(v.clone()) {
            return Ok(path);
        }
        // Try OwnedValue → String → OwnedObjectPath.
        if let Ok(s) = <String>::try_from(v.clone()) {
            return OwnedObjectPath::try_from(s.clone())
                .map_err(|e| CaptureError::Protocol(format!("invalid session_handle '{s}': {e}")));
        }
        // Last resort: extract path from Debug representation.
        // Debug format: OwnedValue(Str("/org/.../session")) or ObjectPath("/org/.../session")
        let debug = format!("{v:?}");
        if let Some(path_str) = extract_quoted_path(&debug) {
            return OwnedObjectPath::try_from(path_str.to_string()).map_err(|e| {
                CaptureError::Protocol(format!("invalid session_handle '{path_str}': {e}"))
            });
        }
        return Err(CaptureError::Protocol(format!(
            "cannot parse session_handle from: {debug}"
        )));
    }

    // Construct deterministically from token (per portal spec).
    let sender = get_sender_name(connection);
    OwnedObjectPath::try_from(format!(
        "/org/freedesktop/portal/desktop/session/{sender}/remoteway_session"
    ))
    .map_err(|e| CaptureError::Protocol(format!("session path construction failed: {e}")))
}

/// Extract a quoted string that looks like an object path from a Debug representation.
/// Looks for `"/org/..."`  pattern inside the debug output.
fn extract_quoted_path(debug: &str) -> Option<&str> {
    let start = debug.find("\"/org/")?;
    let inner = &debug[start + 1..]; // skip opening quote
    let end = inner.find('"')?;
    Some(&inner[..end])
}

/// Extract PipeWire node_id from the Start Response results.
fn extract_pw_node_id(results: &HashMap<String, OwnedValue>) -> Result<u32, CaptureError> {
    let streams = results
        .get("streams")
        .ok_or_else(|| CaptureError::Protocol("missing 'streams' in Start response".into()))?;

    // streams is a(ua{sv}). Parse via Debug representation as the exact
    // zvariant structure type is complex. We look for the first U32 value.
    let streams_str = format!("{streams:?}");
    tracing::info!(streams_raw = %streams_str, "portal Start streams response");
    parse_node_id_from_streams(&streams_str).ok_or_else(|| {
        CaptureError::Protocol(format!(
            "cannot extract node_id from streams: {streams_str}"
        ))
    })
}

/// Parse PipeWire node_id from the debug string of zvariant streams value.
/// The format is typically: `Value(Array([Structure([U32(42), Dict(...)]), ...]))`
/// We look for the first U32 value.
fn parse_node_id_from_streams(s: &str) -> Option<u32> {
    let marker = "U32(";
    let start = s.find(marker)? + marker.len();
    let end = s[start..].find(')')? + start;
    s[start..end].parse().ok()
}

/// Extract stream size from the Start response.
/// The streams dict contains `"size": (I32(w), I32(h))`.
fn extract_stream_size(results: &HashMap<String, OwnedValue>) -> Option<(u32, u32)> {
    let streams = results.get("streams")?;
    let s = format!("{streams:?}");
    parse_stream_size(&s)
}

/// Parse width and height from the streams debug string.
/// Looks for `Str("size"): Value(Structure(... [I32(w), I32(h)] ...))`.
fn parse_stream_size(s: &str) -> Option<(u32, u32)> {
    let size_marker = "\"size\"): Value(Structure";
    let idx = s.find(size_marker)?;
    let after = &s[idx..];

    // Find the I32 values after "size"
    let w = parse_next_i32(after)?;
    let rest = &after[after.find("I32(")? + 4..];
    let rest = &rest[rest.find(')')? + 1..];
    let h = parse_next_i32(rest)?;

    Some((w as u32, h as u32))
}

fn parse_next_i32(s: &str) -> Option<i32> {
    let marker = "I32(";
    let start = s.find(marker)? + marker.len();
    let end = s[start..].find(')')? + start;
    s[start..end].parse().ok()
}

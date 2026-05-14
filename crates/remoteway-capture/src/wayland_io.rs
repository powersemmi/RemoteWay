//! Wayland event-loop helpers shared between capture backends.
//!
//! Provides [`dispatch_with_deadline`] — a non-blocking variant of
//! `EventQueue::blocking_dispatch` that returns after a timeout instead of
//! hanging forever when the compositor stops sending events.

use std::os::fd::AsRawFd;
use std::time::{Duration, Instant};

use wayland_client::{Connection, DispatchError, EventQueue};

use crate::error::CaptureError;

/// Status returned by [`dispatch_with_deadline`].
pub enum DispatchOutcome {
    /// At least one event was dispatched; caller should re-check its state.
    Dispatched,
    /// `deadline` was hit with no events ready. Treat as a soft failure.
    TimedOut,
}

/// Dispatch pending events on `event_queue` and, if none are pending, wait up
/// to `timeout` for the compositor to send some.
///
/// Equivalent to `event_queue.blocking_dispatch(state)` but bounded by
/// `timeout`. Returns `TimedOut` when no events arrived within the window so
/// the caller can decide to abort instead of hanging forever.
pub fn dispatch_with_deadline<S>(
    conn: &Connection,
    event_queue: &mut EventQueue<S>,
    state: &mut S,
    timeout: Duration,
) -> Result<DispatchOutcome, CaptureError> {
    let deadline = Instant::now() + timeout;

    loop {
        // Drain anything already queued before going to the socket.
        let n = event_queue
            .dispatch_pending(state)
            .map_err(map_dispatch_err)?;
        if n > 0 {
            return Ok(DispatchOutcome::Dispatched);
        }

        // Flush outgoing requests so the compositor can answer.
        if let Err(e) = conn.flush() {
            return Err(CaptureError::CaptureFailed(format!("wl flush: {e}")));
        }

        // `prepare_read` returns None if events appeared between
        // `dispatch_pending` and now — loop back and drain again.
        let Some(read_guard) = event_queue.prepare_read() else {
            continue;
        };

        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            // Drop the guard explicitly to release the read lock.
            drop(read_guard);
            return Ok(DispatchOutcome::TimedOut);
        }

        let fd = read_guard.connection_fd().as_raw_fd();
        let timeout_ms: libc::c_int = remaining
            .as_millis()
            .try_into()
            .unwrap_or(libc::c_int::MAX);

        let mut pfd = libc::pollfd {
            fd,
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: pollfd is a valid pointer to a single struct; nfds=1.
        let rc = unsafe { libc::poll(&mut pfd, 1, timeout_ms) };
        if rc < 0 {
            // SAFETY: reading errno after a failed syscall is safe.
            let errno = unsafe { *libc::__errno_location() };
            if errno == libc::EINTR {
                // Signal interrupted us — restart the loop without consuming the deadline.
                drop(read_guard);
                continue;
            }
            return Err(CaptureError::CaptureFailed(format!(
                "poll on wayland fd: errno {errno}"
            )));
        }

        if rc == 0 {
            drop(read_guard);
            return Ok(DispatchOutcome::TimedOut);
        }

        // Socket readable — pull events from the kernel.
        if let Err(e) = read_guard.read() {
            return Err(CaptureError::CaptureFailed(format!("wl read: {e}")));
        }

        // Loop back to dispatch_pending; it will surface the events to the
        // state's Dispatch impls.
    }
}

fn map_dispatch_err(e: DispatchError) -> CaptureError {
    CaptureError::WaylandDispatch(e)
}

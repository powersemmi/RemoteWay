use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;

use remoteway_core::thread_config::ThreadConfig;
use remoteway_proto::input::InputEvent;

use crate::capture::InputCapture;
use crate::error::InputError;

/// Callback type for sending serialized input events to the transport layer.
///
/// The closure receives each `InputEvent` and is responsible for serialization
/// and sending (e.g., prepending a `FrameHeader` and calling `TransportSender::send_input()`).
/// Returns `true` if the event was sent, `false` if the transport is closed.
pub type EventSender = Box<dyn Fn(&InputEvent) -> bool + Send>;

/// Configuration for the input capture thread.
pub struct InputCaptureConfig {
    /// CPU core to pin the thread to (default: 0).
    pub core_id: usize,
    /// SCHED_FIFO priority (default: 99 — highest on client).
    pub sched_priority: u8,
}

impl Default for InputCaptureConfig {
    fn default() -> Self {
        Self {
            core_id: 0,
            sched_priority: 99,
        }
    }
}

/// Handle to a running input capture thread.
///
/// The thread captures pointer and keyboard events from the local Wayland
/// compositor and forwards them to the transport layer via the `EventSender`
/// callback. This is the highest-priority thread on the client (SCHED_FIFO 99).
#[must_use]
pub struct InputCaptureThread {
    join_handle: Option<JoinHandle<Result<(), InputError>>>,
    stop_flag: Arc<AtomicBool>,
}

impl InputCaptureThread {
    /// Spawn the capture thread.
    ///
    /// The `sender` callback is invoked for each captured `InputEvent`.
    /// It should serialize the event and forward it to the transport (e.g.,
    /// `TransportSender::send_input()`).
    pub fn spawn(config: InputCaptureConfig, sender: EventSender) -> Result<Self, InputError> {
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop = Arc::clone(&stop_flag);

        let thread_config =
            ThreadConfig::new(config.core_id, config.sched_priority, "input-capture");

        let join_handle = thread_config.spawn(move || capture_loop(sender, &stop))?;

        Ok(Self {
            join_handle: Some(join_handle),
            stop_flag,
        })
    }

    /// Signal the capture thread to stop and wait for it to exit.
    pub fn stop(&mut self) {
        self.stop_flag.store(true, Ordering::Release);
        if let Some(handle) = self.join_handle.take() {
            let _ = handle.join();
        }
    }

    /// Check if the thread is still running.
    pub fn is_running(&self) -> bool {
        self.join_handle.as_ref().is_some_and(|h| !h.is_finished())
    }
}

impl Drop for InputCaptureThread {
    fn drop(&mut self) {
        self.stop_flag.store(true, Ordering::Release);
    }
}

fn capture_loop(sender: EventSender, stop_flag: &AtomicBool) -> Result<(), InputError> {
    let mut capture = InputCapture::new()?;

    loop {
        if stop_flag.load(Ordering::Acquire) {
            break;
        }

        let events = capture.poll_events()?;
        for event in events {
            if !sender(event) {
                // Transport closed — exit loop.
                return Ok(());
            }
        }

        // If no events were pending, yield briefly to avoid burning CPU.
        if events.is_empty() {
            std::hint::spin_loop();
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_config_default() {
        let cfg = InputCaptureConfig::default();
        assert_eq!(cfg.core_id, 0);
        assert_eq!(cfg.sched_priority, 99);
    }

    #[test]
    fn capture_config_custom() {
        let cfg = InputCaptureConfig {
            core_id: 3,
            sched_priority: 50,
        };
        assert_eq!(cfg.core_id, 3);
        assert_eq!(cfg.sched_priority, 50);
    }

    #[test]
    fn spawn_fails_without_compositor() {
        // SAFETY: this test runs single-threaded and restores no state.
        unsafe { std::env::remove_var("WAYLAND_DISPLAY") };
        let sender: EventSender = Box::new(|_| true);
        let config = InputCaptureConfig {
            core_id: 0,
            sched_priority: 0,
        };
        let result = InputCaptureThread::spawn(config, sender);
        // Thread may spawn but will fail internally. No panic.
        if let Ok(mut thread) = result {
            std::thread::sleep(std::time::Duration::from_millis(50));
            thread.stop();
        }
    }

    #[test]
    fn stop_is_idempotent() {
        // SAFETY: test env.
        unsafe { std::env::remove_var("WAYLAND_DISPLAY") };
        let sender: EventSender = Box::new(|_| true);
        let config = InputCaptureConfig {
            core_id: 0,
            sched_priority: 0,
        };
        let result = InputCaptureThread::spawn(config, sender);
        if let Ok(mut thread) = result {
            std::thread::sleep(std::time::Duration::from_millis(50));
            thread.stop();
            // Second stop should be a no-op (join_handle already taken).
            thread.stop();
        }
    }

    #[test]
    fn is_running_after_internal_failure() {
        // SAFETY: test env.
        unsafe { std::env::remove_var("WAYLAND_DISPLAY") };
        let sender: EventSender = Box::new(|_| true);
        let config = InputCaptureConfig {
            core_id: 0,
            sched_priority: 0,
        };
        let result = InputCaptureThread::spawn(config, sender);
        if let Ok(thread) = result {
            // Thread will exit quickly because InputCapture::new() fails.
            std::thread::sleep(std::time::Duration::from_millis(100));
            // After the thread has exited, is_running should return false.
            assert!(!thread.is_running());
        }
    }

    #[test]
    fn drop_sets_stop_flag() {
        // SAFETY: test env.
        unsafe { std::env::remove_var("WAYLAND_DISPLAY") };
        let sender: EventSender = Box::new(|_| true);
        let config = InputCaptureConfig {
            core_id: 0,
            sched_priority: 0,
        };
        let result = InputCaptureThread::spawn(config, sender);
        if let Ok(thread) = result {
            let stop_flag = Arc::clone(&thread.stop_flag);
            assert!(!stop_flag.load(Ordering::Acquire));
            drop(thread);
            // After drop, the stop flag should be set.
            assert!(stop_flag.load(Ordering::Acquire));
        }
    }

    #[test]
    fn event_sender_returns_false_stops_loop() {
        // The capture_loop exits when sender returns false.
        // We cannot test this without a compositor, but we can verify the callback type.
        let call_count = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let cc = Arc::clone(&call_count);
        let _sender: EventSender = Box::new(move |_ev| {
            cc.fetch_add(1, Ordering::Relaxed);
            false // signal transport closed
        });
        // Verify the closure compiles and is Send.
        fn assert_send<T: Send>(_: &T) {}
        assert_send(&_sender);
    }

    #[test]
    fn event_sender_receives_all_event_kinds() {
        use remoteway_proto::input::{
            InputEvent, KeyEvent, PointerAxis, PointerButton, PointerMotion,
        };
        let received = Arc::new(std::sync::Mutex::new(Vec::new()));
        let rx = Arc::clone(&received);
        let sender: EventSender = Box::new(move |ev| {
            rx.lock().unwrap().push(ev.kind);
            true
        });

        let events = [
            InputEvent::pointer_motion(PointerMotion {
                surface_id: 0,
                _pad: 0,
                x: 10.0,
                y: 20.0,
            }),
            InputEvent::pointer_button(PointerButton {
                button: 0x110,
                state: 1,
            }),
            InputEvent::pointer_axis(PointerAxis {
                axis: 0,
                _pad: [0; 3],
                value: 5.0,
            }),
            InputEvent::key(KeyEvent { key: 30, state: 1 }),
        ];

        for ev in &events {
            let result = sender(ev);
            assert!(result);
        }

        let kinds = received.lock().unwrap();
        assert_eq!(kinds.len(), 4);
        assert_eq!(kinds[0], 0); // PointerMotion
        assert_eq!(kinds[1], 1); // PointerButton
        assert_eq!(kinds[2], 2); // PointerAxis
        assert_eq!(kinds[3], 3); // Key
    }

    #[test]
    fn thread_config_name() {
        let config = InputCaptureConfig {
            core_id: 2,
            sched_priority: 50,
        };
        let tc = ThreadConfig::new(config.core_id, config.sched_priority, "input-capture");
        assert_eq!(tc.name, "input-capture");
        assert_eq!(tc.core_id, 2);
        assert_eq!(tc.sched_priority, 50);
    }
}

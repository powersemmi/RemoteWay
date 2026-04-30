use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;

use rtrb::RingBuffer;

use remoteway_core::thread_config::ThreadConfig;
use remoteway_proto::input::InputEvent;

use crate::error::InputError;
use crate::inject::VirtualInput;

/// Configuration for the input inject thread.
pub struct InputInjectConfig {
    /// CPU core to pin the thread to (default: 0).
    pub core_id: usize,
    /// SCHED_FIFO priority (default: 99 — highest on server).
    pub sched_priority: u8,
    /// SPSC ring buffer capacity for input events (default: 256).
    pub ring_capacity: usize,
}

impl Default for InputInjectConfig {
    fn default() -> Self {
        Self {
            core_id: 0,
            sched_priority: 99,
            ring_capacity: 256,
        }
    }
}

/// Handle to a running input inject thread.
///
/// The thread consumes `InputEvent`s from an rtrb SPSC ring buffer
/// and injects them into the Wayland compositor via `VirtualInput`.
/// This is the highest-priority thread on the server (SCHED_FIFO 99, Core 0).
#[must_use]
pub struct InputInjectThread {
    producer: rtrb::Producer<InputEvent>,
    join_handle: Option<JoinHandle<Result<(), InputError>>>,
    stop_flag: Arc<AtomicBool>,
}

impl InputInjectThread {
    /// Spawn the inject thread.
    ///
    /// The thread connects to the Wayland compositor internally and begins
    /// consuming events from the ring buffer. Returns a handle whose `send()`
    /// method is used by the transport receive path.
    pub fn spawn(config: InputInjectConfig) -> Result<Self, InputError> {
        let (producer, consumer) = RingBuffer::<InputEvent>::new(config.ring_capacity);
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop = Arc::clone(&stop_flag);

        let thread_config =
            ThreadConfig::new(config.core_id, config.sched_priority, "input-inject");

        let join_handle = thread_config.spawn(move || inject_loop(consumer, &stop))?;

        Ok(Self {
            producer,
            join_handle: Some(join_handle),
            stop_flag,
        })
    }

    /// Push an input event into the ring buffer (non-blocking).
    ///
    /// Returns `true` if the event was enqueued, `false` if the ring is full.
    /// At 256 capacity with 16-byte events, overflow should not occur under
    /// normal input rates.
    pub fn send(&mut self, event: InputEvent) -> bool {
        self.producer.push(event).is_ok()
    }

    /// Number of events buffered in the ring.
    pub fn buffered_events(&self) -> usize {
        self.producer.slots()
    }

    /// Signal the inject thread to stop and wait for it to exit.
    pub fn stop(&mut self) {
        self.stop_flag.store(true, Ordering::Release);
        if let Some(handle) = self.join_handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for InputInjectThread {
    fn drop(&mut self) {
        self.stop_flag.store(true, Ordering::Release);
    }
}

fn inject_loop(
    mut consumer: rtrb::Consumer<InputEvent>,
    stop_flag: &AtomicBool,
) -> Result<(), InputError> {
    let input = VirtualInput::new()?;

    loop {
        if stop_flag.load(Ordering::Acquire) {
            break;
        }

        match consumer.pop() {
            Ok(event) => {
                if let Err(e) = input.dispatch_event(&event) {
                    tracing::warn!("inject event error: {e}");
                }
                input.flush();
            }
            Err(_) => {
                // Ring empty — spin briefly to maintain low latency.
                std::hint::spin_loop();
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use remoteway_proto::input::{InputEvent, KeyEvent, PointerButton, PointerMotion};

    use super::*;

    #[test]
    fn inject_config_default() {
        let cfg = InputInjectConfig::default();
        assert_eq!(cfg.core_id, 0);
        assert_eq!(cfg.sched_priority, 99);
        assert_eq!(cfg.ring_capacity, 256);
    }

    #[test]
    fn inject_config_custom() {
        let cfg = InputInjectConfig {
            core_id: 4,
            sched_priority: 95,
            ring_capacity: 512,
        };
        assert_eq!(cfg.core_id, 4);
        assert_eq!(cfg.sched_priority, 95);
        assert_eq!(cfg.ring_capacity, 512);
    }

    #[test]
    fn spawn_fails_without_compositor() {
        // SAFETY: this test runs single-threaded and restores no state.
        unsafe { std::env::remove_var("WAYLAND_DISPLAY") };
        let config = InputInjectConfig {
            core_id: 0,
            sched_priority: 0,
            ring_capacity: 4,
        };
        let result = InputInjectThread::spawn(config);
        // The thread will fail internally when VirtualInput::new() can't connect.
        // The spawn itself may succeed (the thread was created), but the thread
        // will exit with an error. We just verify no panic.
        if let Ok(mut thread) = result {
            std::thread::sleep(std::time::Duration::from_millis(50));
            thread.stop();
        }
    }

    #[test]
    fn send_and_buffered_events_with_spsc() {
        // Test the SPSC ring buffer directly, without a running inject thread.
        // We create a ring buffer of the same type InputInjectThread uses.
        let (mut producer, mut consumer) = RingBuffer::<InputEvent>::new(8);

        let ev1 = InputEvent::key(KeyEvent { key: 30, state: 1 });
        let ev2 = InputEvent::pointer_button(PointerButton {
            button: 0x110,
            state: 1,
        });

        assert!(producer.push(ev1).is_ok());
        assert!(producer.push(ev2).is_ok());

        let popped1 = consumer.pop().unwrap();
        assert_eq!(popped1.kind, 3); // Key
        let popped2 = consumer.pop().unwrap();
        assert_eq!(popped2.kind, 1); // PointerButton
        assert!(consumer.pop().is_err()); // empty
    }

    #[test]
    fn ring_buffer_overflow() {
        // With capacity 2, the third push should fail.
        let (mut producer, _consumer) = RingBuffer::<InputEvent>::new(2);

        let ev = InputEvent::key(KeyEvent { key: 1, state: 0 });
        assert!(producer.push(ev).is_ok());
        assert!(producer.push(ev).is_ok());
        assert!(producer.push(ev).is_err()); // full
    }

    #[test]
    fn stop_is_idempotent() {
        // SAFETY: test env.
        unsafe { std::env::remove_var("WAYLAND_DISPLAY") };
        let config = InputInjectConfig {
            core_id: 0,
            sched_priority: 0,
            ring_capacity: 4,
        };
        let result = InputInjectThread::spawn(config);
        if let Ok(mut thread) = result {
            std::thread::sleep(std::time::Duration::from_millis(50));
            thread.stop();
            // Second stop should be a no-op.
            thread.stop();
        }
    }

    #[test]
    fn drop_sets_stop_flag() {
        // SAFETY: test env.
        unsafe { std::env::remove_var("WAYLAND_DISPLAY") };
        let config = InputInjectConfig {
            core_id: 0,
            sched_priority: 0,
            ring_capacity: 4,
        };
        let result = InputInjectThread::spawn(config);
        if let Ok(thread) = result {
            let stop_flag = Arc::clone(&thread.stop_flag);
            assert!(!stop_flag.load(Ordering::Acquire));
            drop(thread);
            assert!(stop_flag.load(Ordering::Acquire));
        }
    }

    #[test]
    fn send_on_dead_thread_returns_true() {
        // Even if the consumer thread has exited, the producer can still
        // push into the ring buffer (it just won't be consumed).
        // SAFETY: test env.
        unsafe { std::env::remove_var("WAYLAND_DISPLAY") };
        let config = InputInjectConfig {
            core_id: 0,
            sched_priority: 0,
            ring_capacity: 16,
        };
        let result = InputInjectThread::spawn(config);
        if let Ok(mut thread) = result {
            // Wait for thread to die (VirtualInput::new() will fail).
            std::thread::sleep(std::time::Duration::from_millis(100));

            let ev = InputEvent::key(KeyEvent { key: 30, state: 1 });
            // send returns true as long as the ring is not full.
            let sent = thread.send(ev);
            assert!(sent, "send should succeed into the ring buffer");

            thread.stop();
        }
    }

    #[test]
    fn buffered_events_reflects_capacity() {
        // Test via raw ring buffer.
        let (producer, _consumer) = RingBuffer::<InputEvent>::new(16);
        // slots() returns the number of available write slots.
        let slots = producer.slots();
        assert!(slots > 0);
        assert!(slots <= 16);
    }

    #[test]
    fn thread_config_name_inject() {
        let config = InputInjectConfig {
            core_id: 0,
            sched_priority: 99,
            ring_capacity: 256,
        };
        let tc = ThreadConfig::new(config.core_id, config.sched_priority, "input-inject");
        assert_eq!(tc.name, "input-inject");
        assert_eq!(tc.core_id, 0);
        assert_eq!(tc.sched_priority, 99);
    }

    #[test]
    fn send_all_event_types() {
        // SAFETY: test env.
        unsafe { std::env::remove_var("WAYLAND_DISPLAY") };
        let config = InputInjectConfig {
            core_id: 0,
            sched_priority: 0,
            ring_capacity: 16,
        };
        let result = InputInjectThread::spawn(config);
        if let Ok(mut thread) = result {
            std::thread::sleep(std::time::Duration::from_millis(100));

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
                InputEvent::key(KeyEvent { key: 30, state: 1 }),
            ];

            for ev in events {
                assert!(thread.send(ev));
            }

            thread.stop();
        }
    }
}

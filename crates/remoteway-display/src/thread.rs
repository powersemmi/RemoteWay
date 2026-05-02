use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;

use rtrb::RingBuffer;

use remoteway_core::thread_config::ThreadConfig;

use crate::error::DisplayError;
use crate::shm::DamageRegion;

/// A frame ready for display on a specific surface.
#[must_use]
pub struct DisplayFrame {
    /// Target surface identifier.
    pub surface_id: u16,
    /// Pixel data (decompressed).
    pub data: Vec<u8>,
    /// Damage regions for partial updates.
    pub damage: Vec<DamageRegion>,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Bytes per row.
    pub stride: u32,
    /// Capture timestamp in nanoseconds.
    pub timestamp_ns: u64,
}

/// Configuration for the display thread.
pub struct DisplayThreadConfig {
    /// CPU core to pin the thread to.
    pub core_id: usize,
    /// `SCHED_FIFO` priority (0 = default scheduler, typically low for display).
    pub sched_priority: u8,
    /// SPSC ring buffer capacity for incoming frames.
    pub ring_capacity: usize,
}

impl Default for DisplayThreadConfig {
    fn default() -> Self {
        Self {
            core_id: 3,
            sched_priority: 0,
            ring_capacity: 4,
        }
    }
}

/// Handle to a running display thread.
///
/// The display thread consumes decompressed frames from an rtrb SPSC ring buffer,
/// uploads them to Wayland SHM buffers, and commits them to surfaces synchronized
/// with the compositor via `wl_surface.frame` callbacks.
pub struct DisplayThread {
    producer: rtrb::Producer<DisplayFrame>,
    join_handle: Option<JoinHandle<Result<(), DisplayError>>>,
    stop_flag: Arc<AtomicBool>,
}

impl DisplayThread {
    /// Spawn a display thread with the given configuration.
    ///
    /// The thread will connect to the local Wayland compositor,
    /// create surfaces as needed, and display incoming frames.
    pub fn spawn(config: DisplayThreadConfig) -> Result<Self, DisplayError> {
        let (producer, consumer) = RingBuffer::<DisplayFrame>::new(config.ring_capacity);
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop = Arc::clone(&stop_flag);

        let thread_config =
            ThreadConfig::new(config.core_id, config.sched_priority, "display-thread");

        let join_handle = thread_config.spawn(move || display_loop(consumer, &stop))?;

        Ok(Self {
            producer,
            join_handle: Some(join_handle),
            stop_flag,
        })
    }

    /// Send a frame to the display thread for rendering.
    ///
    /// Returns `true` if the frame was enqueued, `false` if the ring is full
    /// (frame dropped).
    pub fn send_frame(&mut self, frame: DisplayFrame) -> bool {
        self.producer.push(frame).is_ok()
    }

    /// Number of frames buffered in the ring.
    pub fn buffered_frames(&self) -> usize {
        self.producer.slots()
    }

    /// Signal the display thread to stop and wait for it to exit.
    pub fn stop(&mut self) {
        self.stop_flag.store(true, Ordering::Release);
        if let Some(handle) = self.join_handle.take() {
            match handle.join() {
                Ok(result) => {
                    if let Err(ref e) = result {
                        tracing::warn!("display thread exited with error: {e}");
                    }
                }
                Err(_) => {
                    // Thread panicked — the panic payload is opaque.
                    tracing::error!("display thread panicked during shutdown");
                }
            }
        }
    }

    /// Check if the display thread is still running.
    pub fn is_running(&self) -> bool {
        self.join_handle.as_ref().is_some_and(|h| !h.is_finished())
    }
}

impl Drop for DisplayThread {
    fn drop(&mut self) {
        self.stop_flag.store(true, Ordering::Release);
    }
}

/// How long the display loop sleeps when idle (no pending frame and nothing
/// new in the ring). Short enough to keep latency low, long enough to avoid
/// burning a full CPU core.
const IDLE_SLEEP: std::time::Duration = std::time::Duration::from_micros(500);

fn display_loop(
    mut consumer: rtrb::Consumer<DisplayFrame>,
    stop_flag: &AtomicBool,
) -> Result<(), DisplayError> {
    use crate::surface::WaylandDisplay;

    let mut display = WaylandDisplay::new()?;

    // Holds the most recent frame per surface that hasn't been committed yet.
    // When the compositor is busy (frame callback pending / buffer not released),
    // we keep the latest frame here and retry on the next iteration instead of
    // dropping it.
    let mut pending_frame: Option<DisplayFrame> = None;

    loop {
        if stop_flag.load(Ordering::Acquire) {
            break;
        }

        // Flush outgoing requests and dispatch incoming Wayland events
        // (frame callbacks, buffer releases, configures).
        display.dispatch_pending()?;

        // Drain the ring buffer, keeping only the newest frame.
        // Intermediate frames are skipped — the display should always show
        // the most recent state to minimise perceived latency.
        while let Ok(frame) = consumer.pop() {
            pending_frame = Some(frame);
        }

        // Try to present the pending frame.
        if let Some(frame) = pending_frame.take() {
            // Ensure surface exists.
            if display.get_surface(frame.surface_id).is_none() {
                display.create_surface(
                    frame.surface_id,
                    &format!("RemoteWay #{}", frame.surface_id),
                    "remoteway",
                    frame.width,
                    frame.height,
                )?;
            }

            // Handle resize if needed.
            if let Some(s) = display.get_surface(frame.surface_id)
                && (s.width != frame.width || s.height != frame.height)
            {
                display.resize_surface(frame.surface_id, frame.width, frame.height)?;
            }

            // Present the frame; if the compositor isn't ready yet, hold onto
            // it for the next loop iteration.
            match display.present_frame(frame.surface_id, &frame.data, &frame.damage) {
                Ok(true) => {} // committed
                Ok(false) => {
                    // Compositor not ready — keep the frame for retry.
                    pending_frame = Some(frame);
                }
                Err(e) => return Err(e),
            }
        } else {
            // Nothing to do — sleep briefly to avoid busy-spinning.
            std::thread::sleep(IDLE_SLEEP);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_thread_config_default() {
        let cfg = DisplayThreadConfig::default();
        assert_eq!(cfg.core_id, 3);
        assert_eq!(cfg.sched_priority, 0);
        assert_eq!(cfg.ring_capacity, 4);
    }

    #[test]
    fn display_thread_config_custom() {
        let cfg = DisplayThreadConfig {
            core_id: 7,
            sched_priority: 50,
            ring_capacity: 16,
        };
        assert_eq!(cfg.core_id, 7);
        assert_eq!(cfg.sched_priority, 50);
        assert_eq!(cfg.ring_capacity, 16);
    }

    #[test]
    fn display_thread_config_min_ring() {
        let cfg = DisplayThreadConfig {
            core_id: 0,
            sched_priority: 0,
            ring_capacity: 1,
        };
        assert_eq!(cfg.ring_capacity, 1);
    }

    #[test]
    fn display_thread_config_large_ring() {
        let cfg = DisplayThreadConfig {
            core_id: 0,
            sched_priority: 0,
            ring_capacity: 1024,
        };
        assert_eq!(cfg.ring_capacity, 1024);
    }

    #[test]
    fn display_frame_fields() {
        let frame = DisplayFrame {
            surface_id: 1,
            data: vec![0u8; 16],
            damage: vec![DamageRegion::new(0, 0, 2, 2)],
            width: 2,
            height: 2,
            stride: 8,
            timestamp_ns: 12345,
        };
        assert_eq!(frame.surface_id, 1);
        assert_eq!(frame.data.len(), 16);
        assert_eq!(frame.damage.len(), 1);
        assert_eq!(frame.width, 2);
        assert_eq!(frame.height, 2);
        assert_eq!(frame.stride, 8);
        assert_eq!(frame.timestamp_ns, 12345);
    }

    #[test]
    fn display_frame_empty_damage() {
        let frame = DisplayFrame {
            surface_id: 0,
            data: vec![0xFF; 1920 * 1080 * 4],
            damage: vec![],
            width: 1920,
            height: 1080,
            stride: 1920 * 4,
            timestamp_ns: 0,
        };
        assert!(frame.damage.is_empty());
        assert_eq!(frame.data.len(), 1920 * 1080 * 4);
    }

    #[test]
    fn display_frame_multiple_damage() {
        let frame = DisplayFrame {
            surface_id: 2,
            data: vec![0u8; 100 * 100 * 4],
            damage: vec![
                DamageRegion::new(0, 0, 50, 50),
                DamageRegion::new(50, 50, 50, 50),
            ],
            width: 100,
            height: 100,
            stride: 400,
            timestamp_ns: 999,
        };
        assert_eq!(frame.damage.len(), 2);
        assert_eq!(frame.damage[0].pixel_count(), 2500);
        assert_eq!(frame.damage[1].pixel_count(), 2500);
    }

    #[test]
    fn display_frame_4k() {
        let w = 3840u32;
        let h = 2160u32;
        let stride = w * 4;
        let frame = DisplayFrame {
            surface_id: 0,
            data: vec![0u8; (stride * h) as usize],
            damage: vec![DamageRegion::new(0, 0, w, h)],
            width: w,
            height: h,
            stride,
            timestamp_ns: 16_666_667,
        };
        assert_eq!(frame.data.len(), 3840 * 2160 * 4);
        assert_eq!(frame.damage[0].pixel_count(), 3840 * 2160);
    }

    #[test]
    fn display_frame_min_size() {
        let frame = DisplayFrame {
            surface_id: u16::MAX,
            data: vec![0xAB; 4], // 1x1 pixel, 4 bytes
            damage: vec![DamageRegion::new(0, 0, 1, 1)],
            width: 1,
            height: 1,
            stride: 4,
            timestamp_ns: 0,
        };
        assert_eq!(frame.data.len(), 4);
        assert_eq!(frame.surface_id, u16::MAX);
    }

    #[test]
    fn display_frame_timestamp_zero() {
        let frame = DisplayFrame {
            surface_id: 0,
            data: vec![],
            damage: vec![],
            width: 0,
            height: 0,
            stride: 0,
            timestamp_ns: 0,
        };
        assert_eq!(frame.timestamp_ns, 0);
    }

    #[test]
    fn display_frame_timestamp_max() {
        let frame = DisplayFrame {
            surface_id: 0,
            data: vec![],
            damage: vec![],
            width: 0,
            height: 0,
            stride: 0,
            timestamp_ns: u64::MAX,
        };
        assert_eq!(frame.timestamp_ns, u64::MAX);
    }

    /// Test rtrb SPSC ring buffer behavior directly (the underlying mechanism).
    #[test]
    fn rtrb_push_pop_basic() {
        let (mut producer, mut consumer) = RingBuffer::<DisplayFrame>::new(4);

        let frame = DisplayFrame {
            surface_id: 1,
            data: vec![0u8; 64],
            damage: vec![],
            width: 4,
            height: 4,
            stride: 16,
            timestamp_ns: 100,
        };

        assert!(producer.push(frame).is_ok());

        let received = consumer.pop().unwrap();
        assert_eq!(received.surface_id, 1);
        assert_eq!(received.data.len(), 64);
        assert_eq!(received.timestamp_ns, 100);
    }

    /// Ring buffer full behavior: push returns error when full.
    #[test]
    fn rtrb_ring_full() {
        let (mut producer, _consumer) = RingBuffer::<DisplayFrame>::new(2);

        let make_frame = |id: u16| DisplayFrame {
            surface_id: id,
            data: vec![0u8; 16],
            damage: vec![],
            width: 2,
            height: 2,
            stride: 8,
            timestamp_ns: id as u64,
        };

        // Fill the ring.
        assert!(producer.push(make_frame(1)).is_ok());
        assert!(producer.push(make_frame(2)).is_ok());

        // Ring is full — push should fail.
        let result = producer.push(make_frame(3));
        assert!(result.is_err());
    }

    /// Drain ring: only the latest frame matters (display loop behavior).
    #[test]
    fn rtrb_drain_keeps_latest() {
        let (mut producer, mut consumer) = RingBuffer::<DisplayFrame>::new(8);

        let make_frame = |id: u16| DisplayFrame {
            surface_id: id,
            data: vec![0u8; 16],
            damage: vec![],
            width: 2,
            height: 2,
            stride: 8,
            timestamp_ns: id as u64 * 1000,
        };

        // Push multiple frames.
        for i in 0..5 {
            assert!(producer.push(make_frame(i)).is_ok());
        }

        // Drain like the display loop: keep only the newest.
        let mut pending: Option<DisplayFrame> = None;
        while let Ok(frame) = consumer.pop() {
            pending = Some(frame);
        }

        let latest = pending.unwrap();
        assert_eq!(latest.surface_id, 4); // The last frame pushed.
        assert_eq!(latest.timestamp_ns, 4000);
    }

    /// `slots()` reports how many items can be written.
    #[test]
    fn rtrb_slots_tracking() {
        let (mut producer, mut consumer) = RingBuffer::<DisplayFrame>::new(4);

        let initial_slots = producer.slots();
        assert_eq!(initial_slots, 4);

        let frame = DisplayFrame {
            surface_id: 0,
            data: vec![0u8; 4],
            damage: vec![],
            width: 1,
            height: 1,
            stride: 4,
            timestamp_ns: 0,
        };
        assert!(producer.push(frame).is_ok());
        assert_eq!(producer.slots(), 3);

        // Pop frees a slot.
        consumer.pop().unwrap();
        assert_eq!(producer.slots(), 4);
    }

    /// `send_frame` maps to producer.push — returns true if queued, false if full.
    #[test]
    fn send_frame_returns_false_when_full() {
        let (mut producer, _consumer) = RingBuffer::<DisplayFrame>::new(1);

        let make_frame = || DisplayFrame {
            surface_id: 0,
            data: vec![0u8; 4],
            damage: vec![],
            width: 1,
            height: 1,
            stride: 4,
            timestamp_ns: 0,
        };

        // First push: succeeds.
        assert!(producer.push(make_frame()).is_ok());

        // Second push: ring full.
        assert!(producer.push(make_frame()).is_err());
    }

    /// `AtomicBool` stop flag pattern used by `DisplayThread`.
    #[test]
    fn stop_flag_pattern() {
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_clone = Arc::clone(&stop_flag);

        assert!(!stop_flag.load(Ordering::Acquire));

        // Signal stop.
        stop_clone.store(true, Ordering::Release);
        assert!(stop_flag.load(Ordering::Acquire));
    }

    /// The idle sleep constant should be 500 microseconds.
    #[test]
    fn idle_sleep_value() {
        assert_eq!(IDLE_SLEEP, std::time::Duration::from_micros(500));
    }

    /// Multiple surfaces can have frames in the same ring.
    #[test]
    fn multiple_surfaces_in_ring() {
        let (mut producer, mut consumer) = RingBuffer::<DisplayFrame>::new(8);

        for sid in 0..3u16 {
            let frame = DisplayFrame {
                surface_id: sid,
                data: vec![0u8; 64],
                damage: vec![DamageRegion::new(0, 0, 4, 4)],
                width: 4,
                height: 4,
                stride: 16,
                timestamp_ns: sid as u64 * 16_666_667,
            };
            assert!(producer.push(frame).is_ok());
        }

        let f0 = consumer.pop().unwrap();
        assert_eq!(f0.surface_id, 0);
        let f1 = consumer.pop().unwrap();
        assert_eq!(f1.surface_id, 1);
        let f2 = consumer.pop().unwrap();
        assert_eq!(f2.surface_id, 2);
        assert!(consumer.pop().is_err());
    }

    /// `DisplayFrame` with large damage list.
    #[test]
    fn display_frame_many_damage_regions() {
        let damage: Vec<DamageRegion> = (0..100)
            .map(|i| DamageRegion::new(i * 10, i * 10, 10, 10))
            .collect();
        let frame = DisplayFrame {
            surface_id: 0,
            data: vec![0u8; 1000 * 1000 * 4],
            damage,
            width: 1000,
            height: 1000,
            stride: 4000,
            timestamp_ns: 0,
        };
        assert_eq!(frame.damage.len(), 100);
        assert_eq!(frame.damage[99].x, 990);
        assert_eq!(frame.damage[99].y, 990);
    }
}

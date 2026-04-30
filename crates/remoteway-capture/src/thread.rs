use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;

use rtrb::RingBuffer;

use remoteway_core::thread_config::ThreadConfig;

use crate::backend::{CaptureBackend, CapturedFrame};
use crate::error::CaptureError;

/// Configuration for the capture thread.
pub struct CaptureThreadConfig {
    /// CPU core to pin the thread to.
    pub core_id: usize,
    /// SCHED_FIFO priority (typically 90 for capture).
    pub sched_priority: u8,
    /// SPSC ring buffer capacity for frames.
    pub ring_capacity: usize,
}

impl Default for CaptureThreadConfig {
    fn default() -> Self {
        Self {
            core_id: 1,
            sched_priority: 90,
            ring_capacity: 4,
        }
    }
}

/// Handle to a running capture thread.
///
/// The thread captures frames via a [`CaptureBackend`] and pushes them
/// into an rtrb SPSC ring buffer. The consumer reads frames from the
/// other end of the ring buffer (typically the compress stage).
pub struct CaptureThread {
    consumer: rtrb::Consumer<CapturedFrame>,
    join_handle: Option<JoinHandle<Result<(), CaptureError>>>,
    stop_flag: Arc<AtomicBool>,
    /// Set by the capture loop when a frame is dropped because the ring is
    /// full. The compress thread reads and clears this flag so it can expand
    /// damage to the full frame (the compositor's damage regions become
    /// inaccurate relative to the last frame the compress thread processed).
    frames_dropped: Arc<AtomicBool>,
    /// Set by the capture loop just before it exits. The compress thread
    /// checks this to detect capture session end (e.g. toplevel closed)
    /// and trigger server shutdown.
    finished: Arc<AtomicBool>,
}

impl CaptureThread {
    /// Spawn a capture thread with the given backend and configuration.
    pub fn spawn(
        mut backend: Box<dyn CaptureBackend>,
        config: CaptureThreadConfig,
    ) -> Result<Self, CaptureError> {
        let (producer, consumer) = RingBuffer::<CapturedFrame>::new(config.ring_capacity);
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop = Arc::clone(&stop_flag);
        let frames_dropped = Arc::new(AtomicBool::new(false));
        let dropped = Arc::clone(&frames_dropped);
        let finished = Arc::new(AtomicBool::new(false));
        let fin = Arc::clone(&finished);

        let thread_config =
            ThreadConfig::new(config.core_id, config.sched_priority, "capture-thread");

        let join_handle = thread_config
            .spawn(move || capture_loop(&mut *backend, producer, &stop, &dropped, &fin))?;

        Ok(Self {
            consumer,
            join_handle: Some(join_handle),
            stop_flag,
            frames_dropped,
            finished,
        })
    }

    /// Try to read the next captured frame (non-blocking).
    pub fn try_recv(&mut self) -> Option<CapturedFrame> {
        self.consumer.pop().ok()
    }

    /// Number of frames buffered in the ring.
    pub fn buffered_frames(&self) -> usize {
        self.consumer.slots()
    }

    /// Returns `true` if any frames were dropped since the last call, then
    /// clears the flag. The compress thread uses this to force full-frame
    /// damage so the delta base stays in sync with the client.
    pub fn take_dropped_flag(&self) -> bool {
        self.frames_dropped.swap(false, Ordering::AcqRel)
    }

    /// Returns `true` if the capture thread has exited (session ended,
    /// error, or stop requested). The compress thread uses this to detect
    /// that no more frames will arrive and trigger server shutdown.
    pub fn is_finished(&self) -> bool {
        self.finished.load(Ordering::Acquire)
    }

    /// Signal the capture thread to stop and wait for it to exit.
    pub fn stop(&mut self) {
        self.stop_flag.store(true, Ordering::Release);
        if let Some(handle) = self.join_handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for CaptureThread {
    fn drop(&mut self) {
        // Signal the thread to stop. The thread checks stop_flag between frames,
        // so it will exit after the current frame completes.
        self.stop_flag.store(true, Ordering::Release);
        // Drop the consumer (closes the ring buffer), which also signals the thread.
        // We detach the thread — it will exit on its own after the next next_frame()
        // returns or the stop flag is checked. No blocking in Drop.
    }
}

fn capture_loop(
    backend: &mut dyn CaptureBackend,
    mut producer: rtrb::Producer<CapturedFrame>,
    stop_flag: &AtomicBool,
    frames_dropped: &AtomicBool,
    finished: &AtomicBool,
) -> Result<(), CaptureError> {
    let result = (|| {
        loop {
            if stop_flag.load(Ordering::Acquire) {
                backend.stop();
                break;
            }

            let frame = match backend.next_frame() {
                Ok(f) => f,
                Err(CaptureError::SessionEnded) => {
                    tracing::info!("capture session ended (toplevel closed or output removed)");
                    break;
                }
                Err(e) => return Err(e),
            };

            // Try to push the frame. If ring is full, drop it and signal the
            // compress thread that damage regions may be inaccurate.
            if producer.push(frame).is_err() {
                tracing::warn!("capture ring full, dropping frame");
                frames_dropped.store(true, Ordering::Release);
            }
        }

        Ok(())
    })();

    finished.store(true, Ordering::Release);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_thread_config_default() {
        let cfg = CaptureThreadConfig::default();
        assert_eq!(cfg.core_id, 1);
        assert_eq!(cfg.sched_priority, 90);
        assert_eq!(cfg.ring_capacity, 4);
    }

    struct MockBackend {
        frames: Vec<CapturedFrame>,
        idx: usize,
    }

    impl CaptureBackend for MockBackend {
        fn next_frame(&mut self) -> Result<CapturedFrame, CaptureError> {
            if self.idx >= self.frames.len() {
                return Err(CaptureError::SessionEnded);
            }
            let frame = CapturedFrame {
                data: self.frames[self.idx].data.clone(),
                damage: self.frames[self.idx].damage.clone(),
                format: self.frames[self.idx].format,
                width: self.frames[self.idx].width,
                height: self.frames[self.idx].height,
                stride: self.frames[self.idx].stride,
                timestamp_ns: self.frames[self.idx].timestamp_ns,
            };
            self.idx += 1;
            Ok(frame)
        }

        fn name(&self) -> &'static str {
            "mock"
        }

        fn stop(&mut self) {}
    }

    fn make_test_frame(ts: u64) -> CapturedFrame {
        CapturedFrame {
            data: vec![0u8; 16],
            damage: vec![remoteway_compress::delta::DamageRect::new(0, 0, 2, 2)],
            format: crate::backend::PixelFormat::Xrgb8888,
            width: 2,
            height: 2,
            stride: 8,
            timestamp_ns: ts,
        }
    }

    #[test]
    fn capture_thread_receives_frames() {
        let frames = vec![make_test_frame(1), make_test_frame(2), make_test_frame(3)];
        let backend = Box::new(MockBackend { frames, idx: 0 });

        let config = CaptureThreadConfig {
            core_id: 0,
            sched_priority: 0,
            ring_capacity: 8,
        };

        let mut thread = CaptureThread::spawn(backend, config).unwrap();

        // Wait for the thread to produce frames.
        std::thread::sleep(std::time::Duration::from_millis(100));

        let mut received = Vec::new();
        while let Some(frame) = thread.try_recv() {
            received.push(frame.timestamp_ns);
        }
        assert_eq!(received, vec![1, 2, 3]);
    }

    #[test]
    fn stop_flag_terminates_thread() {
        struct InfiniteBackend;
        impl CaptureBackend for InfiniteBackend {
            fn next_frame(&mut self) -> Result<CapturedFrame, CaptureError> {
                std::thread::sleep(std::time::Duration::from_millis(10));
                Ok(make_test_frame(0))
            }
            fn name(&self) -> &'static str {
                "infinite"
            }
            fn stop(&mut self) {}
        }

        let config = CaptureThreadConfig {
            core_id: 0,
            sched_priority: 0,
            ring_capacity: 4,
        };
        let mut thread = CaptureThread::spawn(Box::new(InfiniteBackend), config).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(50));
        thread.stop();
        // If stop didn't work, this test would hang forever.
    }

    #[test]
    fn error_propagation_from_backend() {
        struct FailBackend;
        impl CaptureBackend for FailBackend {
            fn next_frame(&mut self) -> Result<CapturedFrame, CaptureError> {
                Err(CaptureError::CaptureFailed("test error".into()))
            }
            fn name(&self) -> &'static str {
                "fail"
            }
            fn stop(&mut self) {}
        }

        let config = CaptureThreadConfig {
            core_id: 0,
            sched_priority: 0,
            ring_capacity: 4,
        };
        let mut thread = CaptureThread::spawn(Box::new(FailBackend), config).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(50));
        // Thread should have exited; no frames available.
        assert!(thread.try_recv().is_none());
    }
}

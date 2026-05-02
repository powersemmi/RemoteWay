//! Integration tests for remoteway-display without a real Wayland compositor.

use remoteway_display::shm::DamageRegion;
use remoteway_display::thread::{DisplayFrame, DisplayThreadConfig};
use remoteway_display::{DisplayError, WaylandDisplay};

#[test]
fn display_error_variants_display() {
    let errors: Vec<DisplayError> = vec![
        DisplayError::NoCompositor,
        DisplayError::NoShm,
        DisplayError::NoXdgWmBase,
        DisplayError::NoSeat,
        DisplayError::ShmBuffer("test error".into()),
        DisplayError::NotConfigured,
        DisplayError::SurfaceNotFound(99),
        DisplayError::SessionEnded,
    ];
    for e in &errors {
        let s = format!("{e}");
        assert!(!s.is_empty());
    }
}

#[test]
fn display_error_debug() {
    let e = DisplayError::SurfaceNotFound(42);
    let dbg = format!("{e:?}");
    assert!(dbg.contains("SurfaceNotFound"));
    assert!(dbg.contains("42"));
}

#[test]
fn wayland_display_fails_without_display() {
    // SAFETY: removing env var in test setup is safe — this is the only test using it.
    unsafe { std::env::remove_var("WAYLAND_DISPLAY") };
    let result = WaylandDisplay::new();
    assert!(result.is_err());
}

#[test]
fn display_thread_config_custom() {
    let cfg = DisplayThreadConfig {
        core_id: 5,
        sched_priority: 10,
        ring_capacity: 16,
    };
    assert_eq!(cfg.core_id, 5);
    assert_eq!(cfg.sched_priority, 10);
    assert_eq!(cfg.ring_capacity, 16);
}

#[test]
fn display_frame_4k_full_damage() {
    let w = 3840u32;
    let h = 2160u32;
    let stride = w * 4;
    let frame = DisplayFrame {
        surface_id: 0,
        data: vec![0xAB; (stride * h) as usize],
        damage: vec![],
        width: w,
        height: h,
        stride,
        timestamp_ns: 1_000_000,
    };
    assert_eq!(frame.data.len(), 3840 * 2160 * 4);
    assert!(frame.damage.is_empty());
}

#[test]
fn display_frame_partial_damage() {
    let w = 1920u32;
    let h = 1080u32;
    let stride = w * 4;
    let total = (stride * h) as usize;
    let damage = vec![
        DamageRegion::new(100, 200, 300, 100),
        DamageRegion::new(500, 500, 200, 200),
        DamageRegion::new(0, 0, 50, 50),
    ];
    let frame = DisplayFrame {
        surface_id: 1,
        data: vec![0xFF; total],
        damage: damage.clone(),
        width: w,
        height: h,
        stride,
        timestamp_ns: 42,
    };
    assert_eq!(frame.damage.len(), 3);
    assert_eq!(frame.damage[0].pixel_count(), 300 * 100);
    assert_eq!(frame.damage[1].pixel_count(), 200 * 200);
}

#[test]
fn damage_region_equality() {
    let a = DamageRegion::new(10, 20, 100, 50);
    let b = DamageRegion::new(10, 20, 100, 50);
    let c = DamageRegion::new(10, 20, 100, 51);
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn damage_region_debug_format() {
    let r = DamageRegion::new(1, 2, 3, 4);
    let dbg = format!("{r:?}");
    assert!(dbg.contains("DamageRegion"));
    assert!(dbg.contains("1"));
    assert!(dbg.contains("2"));
    assert!(dbg.contains("3"));
    assert!(dbg.contains("4"));
}

#[test]
fn damage_region_clone() {
    let r = DamageRegion::new(10, 20, 30, 40);
    let c = r;
    assert_eq!(r, c);
}

#[test]
fn display_frame_multiple_surfaces() {
    let frames: Vec<DisplayFrame> = (0..5)
        .map(|id| DisplayFrame {
            surface_id: id,
            data: vec![0u8; 64 * 48 * 4],
            damage: vec![DamageRegion::new(0, 0, 64, 48)],
            width: 64,
            height: 48,
            stride: 64 * 4,
            timestamp_ns: id as u64 * 16_666_667,
        })
        .collect();
    assert_eq!(frames.len(), 5);
    for (i, f) in frames.iter().enumerate() {
        assert_eq!(f.surface_id, i as u16);
    }
}

#[test]
fn display_error_surface_not_found_message() {
    let e = DisplayError::SurfaceNotFound(42);
    let msg = e.to_string();
    assert!(msg.contains("42"));
    assert!(msg.contains("not found"));
}

#[test]
fn display_error_shm_buffer_message() {
    let e = DisplayError::ShmBuffer("stride overflow: width * 4 exceeds usize".into());
    let msg = e.to_string();
    assert!(msg.contains("stride overflow"));
}

#[test]
fn display_error_no_compositor_message() {
    let e = DisplayError::NoCompositor;
    assert!(e.to_string().contains("compositor"));
}

#[test]
fn display_error_no_shm_message() {
    let e = DisplayError::NoShm;
    assert!(e.to_string().contains("shm"));
}

#[test]
fn display_error_no_xdg_wm_base_message() {
    let e = DisplayError::NoXdgWmBase;
    assert!(e.to_string().contains("xdg_wm_base"));
}

#[test]
fn display_error_session_ended_message() {
    let e = DisplayError::SessionEnded;
    assert!(e.to_string().contains("ended"));
}

#[test]
fn display_error_not_configured_message() {
    let e = DisplayError::NotConfigured;
    assert!(e.to_string().contains("configured"));
}

#[test]
fn display_error_no_seat_message() {
    let e = DisplayError::NoSeat;
    assert!(e.to_string().contains("seat"));
}

/// Verify `DamageRegion` can represent full-frame damage for common resolutions.
#[test]
fn damage_region_common_resolutions() {
    let resolutions: Vec<(u32, u32, usize)> = vec![
        (640, 480, 307_200),
        (1280, 720, 921_600),
        (1920, 1080, 2_073_600),
        (2560, 1440, 3_686_400),
        (3840, 2160, 8_294_400),
        (7680, 4320, 33_177_600),
    ];
    for (w, h, expected_pixels) in resolutions {
        let r = DamageRegion::new(0, 0, w, h);
        assert_eq!(
            r.pixel_count(),
            expected_pixels,
            "pixel_count mismatch for {w}x{h}"
        );
    }
}

/// Test that `DisplayThreadConfig` default values are reasonable.
#[test]
fn display_thread_config_defaults_reasonable() {
    let cfg = DisplayThreadConfig::default();
    // Core 3 is typical for a 4-core layout where cores 0-2 handle other stages.
    assert!(cfg.core_id < 256);
    // Priority 0 = default scheduler (display is not the highest priority stage).
    assert_eq!(cfg.sched_priority, 0);
    // Ring capacity 4 gives a small buffer without excessive latency.
    assert!(cfg.ring_capacity >= 1);
    assert!(cfg.ring_capacity <= 64);
}

/// Stress test: create many `DisplayFrame` objects.
#[test]
fn display_frame_batch_creation() {
    let frames: Vec<DisplayFrame> = (0..100u16)
        .map(|i| DisplayFrame {
            surface_id: i % 4,
            data: vec![0u8; 64 * 48 * 4],
            damage: if i % 3 == 0 {
                vec![]
            } else {
                vec![DamageRegion::new(0, 0, 64, 48)]
            },
            width: 64,
            height: 48,
            stride: 256,
            timestamp_ns: i as u64 * 16_666_667,
        })
        .collect();
    assert_eq!(frames.len(), 100);
    // Every third frame has empty damage (full frame).
    let empty_count = frames.iter().filter(|f| f.damage.is_empty()).count();
    assert_eq!(empty_count, 34); // 0, 3, 6, ..., 99 = 34 frames
}

/// Test frame stride consistency.
#[test]
fn display_frame_stride_consistency() {
    let widths: Vec<u32> = vec![1, 2, 64, 640, 1920, 2560, 3840, 7680];
    for w in widths {
        let stride = w * 4;
        let h = 100u32;
        let frame = DisplayFrame {
            surface_id: 0,
            data: vec![0u8; (stride * h) as usize],
            damage: vec![],
            width: w,
            height: h,
            stride,
            timestamp_ns: 0,
        };
        assert_eq!(frame.stride, w * 4);
        assert_eq!(frame.data.len(), (w * 4 * h) as usize);
    }
}

/// Verify `DisplayFrame` is large enough to hold the pixel data.
#[test]
fn display_frame_data_size_matches_dimensions() {
    let frame = DisplayFrame {
        surface_id: 0,
        data: vec![0xCD; 1920 * 1080 * 4],
        damage: vec![DamageRegion::new(100, 200, 500, 300)],
        width: 1920,
        height: 1080,
        stride: 1920 * 4,
        timestamp_ns: 0,
    };
    let expected_size = frame.stride as usize * frame.height as usize;
    assert_eq!(frame.data.len(), expected_size);
}

/// Verify rtrb SPSC behavior when used in integration context.
#[test]
fn rtrb_spsc_integration() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    let (mut producer, mut consumer) = rtrb::RingBuffer::<DisplayFrame>::new(8);
    let done = Arc::new(AtomicBool::new(false));
    let done_clone = Arc::clone(&done);

    let writer = std::thread::spawn(move || {
        for i in 0..20u16 {
            let frame = DisplayFrame {
                surface_id: i,
                data: vec![i as u8; 16],
                damage: vec![],
                width: 2,
                height: 2,
                stride: 8,
                timestamp_ns: i as u64 * 1000,
            };
            // Retry if ring is full.
            loop {
                match producer.push(frame.clone_for_test(i)) {
                    Ok(()) => break,
                    Err(_) => std::thread::yield_now(),
                }
            }
        }
        done_clone.store(true, Ordering::Release);
    });

    let mut received = Vec::new();
    loop {
        if let Ok(frame) = consumer.pop() {
            received.push(frame.surface_id);
        } else if done.load(Ordering::Acquire) {
            // Drain remaining.
            while let Ok(frame) = consumer.pop() {
                received.push(frame.surface_id);
            }
            break;
        } else {
            std::thread::yield_now();
        }
    }

    writer.join().unwrap();
    assert_eq!(received.len(), 20);
    for (i, &sid) in received.iter().enumerate() {
        assert_eq!(sid, i as u16);
    }
}

/// Helper trait for creating test frames in integration tests.
trait CloneForTest {
    fn clone_for_test(&self, id: u16) -> DisplayFrame;
}

impl CloneForTest for DisplayFrame {
    fn clone_for_test(&self, id: u16) -> DisplayFrame {
        DisplayFrame {
            surface_id: id,
            data: self.data.clone(),
            damage: self.damage.clone(),
            width: self.width,
            height: self.height,
            stride: self.stride,
            timestamp_ns: id as u64 * 1000,
        }
    }
}

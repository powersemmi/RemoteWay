//! Integration tests for remoteway-interpolate public API.

use remoteway_interpolate::{
    BackendDetector, BackendKind, FrameInterpolator, GpuFrame, InterpolateError,
    InterpolationManager, LinearBlendInterpolator,
};

fn make_frame(width: u32, height: u32, value: u8, ts: u64) -> GpuFrame {
    let stride = width * 4;
    GpuFrame::from_data(
        vec![value; (stride * height) as usize],
        width,
        height,
        stride,
        ts,
    )
}

// --- GpuFrame ---

#[test]
fn gpu_frame_4k_allocation() {
    let f = GpuFrame::new(3840, 2160, 3840 * 4, 0);
    assert_eq!(f.byte_size(), 3840 * 2160 * 4);
    assert_eq!(f.pixel_count(), 3840 * 2160);
}

#[test]
fn gpu_frame_zero_size() {
    let f = GpuFrame::new(0, 0, 0, 0);
    assert_eq!(f.byte_size(), 0);
    assert_eq!(f.pixel_count(), 0);
}

#[test]
fn gpu_frame_same_dimensions_with_different_stride() {
    let a = GpuFrame::new(100, 50, 400, 0);
    let b = GpuFrame::new(100, 50, 512, 0); // padded stride
    assert!(!a.same_dimensions(&b));
}

// --- LinearBlendInterpolator ---

#[test]
fn linear_blend_full_pipeline() {
    let interp = LinearBlendInterpolator;
    let a = make_frame(320, 240, 0, 0);
    let b = make_frame(320, 240, 200, 16_666_667);

    // Generate 3 intermediate frames.
    for i in 1..=3 {
        let t = i as f32 / 4.0;
        let result = interp.interpolate(&a, &b, t).unwrap();
        assert_eq!(result.width, 320);
        assert_eq!(result.height, 240);
        assert_eq!(result.byte_size(), 320 * 240 * 4);
        assert!(result.timestamp_ns > 0);
        assert!(result.timestamp_ns < 16_666_667);
    }
}

#[test]
fn linear_blend_preserves_extremes() {
    let interp = LinearBlendInterpolator;
    let black = make_frame(8, 8, 0, 0);
    let white = make_frame(8, 8, 255, 1000);

    let at_0 = interp.interpolate(&black, &white, 0.0).unwrap();
    let at_1 = interp.interpolate(&black, &white, 1.0).unwrap();

    for &px in &at_0.data {
        assert!(px <= 1, "at t=0: px={px}");
    }
    for &px in &at_1.data {
        assert!(px >= 254, "at t=1: px={px}");
    }
}

#[test]
fn linear_blend_monotonic() {
    let interp = LinearBlendInterpolator;
    let a = make_frame(4, 4, 0, 0);
    let b = make_frame(4, 4, 200, 1000);

    let mut prev_avg = 0u64;
    for i in 0..=10 {
        let t = i as f32 / 10.0;
        let result = interp.interpolate(&a, &b, t).unwrap();
        let avg: u64 =
            result.data.iter().map(|&x| x as u64).sum::<u64>() / result.data.len() as u64;
        assert!(
            avg >= prev_avg,
            "non-monotonic at t={t}: {avg} < {prev_avg}"
        );
        prev_avg = avg;
    }
}

#[test]
fn linear_blend_timestamp_interpolation() {
    let interp = LinearBlendInterpolator;
    let a = make_frame(2, 2, 0, 1_000_000);
    let b = make_frame(2, 2, 0, 2_000_000);

    let mid = interp.interpolate(&a, &b, 0.5).unwrap();
    assert!(mid.timestamp_ns >= 1_400_000 && mid.timestamp_ns <= 1_600_000);

    let quarter = interp.interpolate(&a, &b, 0.25).unwrap();
    assert!(quarter.timestamp_ns >= 1_200_000 && quarter.timestamp_ns <= 1_300_000);
}

// --- BackendDetector ---

#[test]
fn backend_detector_always_has_fallback() {
    let available = BackendDetector::detect_available();
    assert!(available.contains(&BackendKind::LinearBlend));
    assert_eq!(*available.last().unwrap(), BackendKind::LinearBlend);
}

#[test]
fn backend_detector_select_best_works() {
    let interp = BackendDetector::select_best().unwrap();
    // Without GPU features: linear-blend. With GPU features + GPU: may be a GPU backend.
    assert!(!interp.name().is_empty());

    // Verify it actually works.
    let a = make_frame(16, 16, 50, 0);
    let b = make_frame(16, 16, 150, 1000);
    let result = interp.interpolate(&a, &b, 0.5).unwrap();
    assert_eq!(result.width, 16);
}

#[test]
fn backend_detector_create_linear_blend() {
    assert!(BackendDetector::create_backend(BackendKind::LinearBlend).is_ok());
}

#[test]
fn backend_detector_create_all_returns_result() {
    // Each backend returns either Ok (if available) or a non-empty error.
    for kind in [
        BackendKind::WgpuOpticalFlow,
        BackendKind::Fsr2,
        BackendKind::NvidiaOpticalFlow,
        BackendKind::Fsr3Hardware,
        BackendKind::Rife,
    ] {
        let result = BackendDetector::create_backend(kind);
        match result {
            Ok(interp) => assert!(!interp.name().is_empty(), "empty name for {kind:?}"),
            Err(e) => assert!(!format!("{e}").is_empty(), "empty error for {kind:?}"),
        }
    }
}

// --- InterpolationManager ---

#[test]
fn manager_full_workflow() {
    let mut mgr = InterpolationManager::new(Box::new(LinearBlendInterpolator));

    // Push first frame — cannot interpolate yet.
    mgr.push_frame(make_frame(64, 48, 0, 0));
    assert!(!mgr.can_interpolate());
    assert_eq!(mgr.frame_count(), 1);

    // Push second frame — now we can.
    mgr.push_frame(make_frame(64, 48, 200, 16_666_667));
    assert!(mgr.can_interpolate());
    assert_eq!(mgr.frame_count(), 2);

    // Generate 2 intermediate frames.
    let f1 = mgr.interpolate(0.33).unwrap().unwrap();
    let f2 = mgr.interpolate(0.66).unwrap().unwrap();
    assert_eq!(mgr.interpolated_count(), 2);

    // f2 pixels should be brighter than f1.
    let avg1: u64 = f1.data.iter().map(|&x| x as u64).sum::<u64>() / f1.data.len() as u64;
    let avg2: u64 = f2.data.iter().map(|&x| x as u64).sum::<u64>() / f2.data.len() as u64;
    assert!(avg2 > avg1);

    // Push third frame — window slides.
    mgr.push_frame(make_frame(64, 48, 50, 33_333_334));
    assert_eq!(mgr.anchor_timestamp_ns(), Some(16_666_667));
    assert_eq!(mgr.target_timestamp_ns(), Some(33_333_334));
}

#[test]
fn manager_anchor_reset_on_scene_change() {
    let mut mgr = InterpolationManager::new(Box::new(LinearBlendInterpolator));

    mgr.push_frame(make_frame(64, 48, 0, 0));
    mgr.push_frame(make_frame(64, 48, 100, 1000));
    assert!(mgr.can_interpolate());

    // Simulate scene change.
    mgr.reset_anchor();
    mgr.push_frame(make_frame(64, 48, 200, 2000));
    assert!(!mgr.can_interpolate());

    // Next frame restores pair.
    mgr.push_frame(make_frame(64, 48, 220, 3000));
    assert!(mgr.can_interpolate());
}

#[test]
fn manager_clear() {
    let mut mgr = InterpolationManager::new(Box::new(LinearBlendInterpolator));
    mgr.push_frame(make_frame(64, 48, 0, 0));
    mgr.push_frame(make_frame(64, 48, 100, 1000));
    mgr.interpolate(0.5).unwrap();

    mgr.clear();
    assert!(!mgr.can_interpolate());
    assert_eq!(mgr.frame_count(), 0);
    assert_eq!(mgr.interpolated_count(), 0);
}

#[test]
fn manager_high_frame_count() {
    let mut mgr = InterpolationManager::new(Box::new(LinearBlendInterpolator));
    for i in 0..100u64 {
        mgr.push_frame(make_frame(16, 16, (i % 256) as u8, i * 16_666_667));
    }
    assert_eq!(mgr.frame_count(), 100);
    assert!(mgr.can_interpolate());
}

// --- Error paths ---

#[test]
fn interpolate_error_display() {
    let errors = vec![
        InterpolateError::NoBackend,
        InterpolateError::DeviceLost,
        InterpolateError::DimensionMismatch(1920, 1080, 3840, 2160),
        InterpolateError::InvalidFactor(2.0),
        InterpolateError::InitFailed("gpu init failed".into()),
        InterpolateError::InterpolateFailed("shader error".into()),
    ];
    for e in &errors {
        assert!(!format!("{e}").is_empty());
    }
}

#[test]
fn interpolate_error_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<InterpolateError>();
}

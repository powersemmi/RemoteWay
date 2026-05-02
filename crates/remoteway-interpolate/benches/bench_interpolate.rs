use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use remoteway_interpolate::{FrameInterpolator, GpuFrame, LinearBlendInterpolator};

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

const RESOLUTIONS: &[(u32, u32, &str)] = &[
    (1920, 1080, "1080p"),
    (2560, 1440, "1440p"),
    (3840, 2160, "4K"),
];

// ---------------------------------------------------------------------------
// Linear Blend (CPU)
// ---------------------------------------------------------------------------

fn bench_linear_blend(c: &mut Criterion) {
    let interp = LinearBlendInterpolator;

    let mut group = c.benchmark_group("linear_blend");
    for &(w, h, label) in RESOLUTIONS {
        let a = make_frame(w, h, 50, 0);
        let b = make_frame(w, h, 200, 16_666_667);
        group.bench_with_input(
            BenchmarkId::new("interpolate", label),
            &(a, b),
            |bench, (a, b)| {
                bench.iter(|| interp.interpolate(a, b, 0.5).unwrap());
            },
        );
    }
    group.finish();
}

fn bench_linear_blend_latency(c: &mut Criterion) {
    let interp = LinearBlendInterpolator;
    let a = make_frame(1920, 1080, 0, 0);
    let b = make_frame(1920, 1080, 255, 16_666_667);

    let mut group = c.benchmark_group("linear_blend_latency");
    for t_percent in [25, 50, 75] {
        let t = t_percent as f32 / 100.0;
        group.bench_with_input(BenchmarkId::new("t", t_percent), &t, |bench, &t| {
            bench.iter(|| interp.interpolate(&a, &b, t).unwrap());
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// wgpu Optical Flow (GPU)
// ---------------------------------------------------------------------------

fn bench_wgpu_optical_flow(c: &mut Criterion) {
    #[cfg(feature = "wgpu-backend")]
    {
        use remoteway_interpolate::backends::wgpu_optical_flow::WgpuOpticalFlowInterpolator;

        let interp = match WgpuOpticalFlowInterpolator::new() {
            Ok(i) => i,
            Err(e) => {
                eprintln!("wgpu: skipping — {e}");
                return;
            }
        };

        let mut group = c.benchmark_group("wgpu_optical_flow");
        for &(w, h, label) in RESOLUTIONS {
            let a = make_frame(w, h, 50, 0);
            let b = make_frame(w, h, 200, 16_666_667);
            group.bench_with_input(
                BenchmarkId::new("interpolate", label),
                &(a, b),
                |bench, (a, b)| {
                    bench.iter(|| interp.interpolate(a, b, 0.5).unwrap());
                },
            );
        }
        group.finish();
    }
    #[cfg(not(feature = "wgpu-backend"))]
    {
    }
}

// ---------------------------------------------------------------------------
// FSR2 (Vulkan compute)
// ---------------------------------------------------------------------------

fn bench_fsr2(c: &mut Criterion) {
    #[cfg(feature = "fsr2")]
    {
        use remoteway_interpolate::backends::fsr2::Fsr2Interpolator;

        let interp = match Fsr2Interpolator::new() {
            Ok(i) => i,
            Err(e) => {
                eprintln!("fsr2: skipping — {e}");
                return;
            }
        };

        let mut group = c.benchmark_group("fsr2");
        for &(w, h, label) in RESOLUTIONS {
            let a = make_frame(w, h, 50, 0);
            let b = make_frame(w, h, 200, 16_666_667);
            group.bench_with_input(
                BenchmarkId::new("interpolate", label),
                &(a, b),
                |bench, (a, b)| {
                    bench.iter(|| interp.interpolate(a, b, 0.5).unwrap());
                },
            );
        }
        group.finish();
    }
    #[cfg(not(feature = "fsr2"))]
    {
    }
}

// ---------------------------------------------------------------------------
// FSR3 (Vulkan compute, RDNA3+ tuned)
// ---------------------------------------------------------------------------

fn bench_fsr3(c: &mut Criterion) {
    #[cfg(feature = "fsr3")]
    {
        use remoteway_interpolate::backends::fsr3::Fsr3Interpolator;

        let interp = match Fsr3Interpolator::new() {
            Ok(i) => i,
            Err(e) => {
                eprintln!("fsr3: skipping — {e}");
                return;
            }
        };

        let mut group = c.benchmark_group("fsr3");
        for &(w, h, label) in RESOLUTIONS {
            let a = make_frame(w, h, 50, 0);
            let b = make_frame(w, h, 200, 16_666_667);
            group.bench_with_input(
                BenchmarkId::new("interpolate", label),
                &(a, b),
                |bench, (a, b)| {
                    bench.iter(|| interp.interpolate(a, b, 0.5).unwrap());
                },
            );
        }
        group.finish();
    }
    #[cfg(not(feature = "fsr3"))]
    {
    }
}

// ---------------------------------------------------------------------------
// NVIDIA Optical Flow
// ---------------------------------------------------------------------------

fn bench_nvidia_of(c: &mut Criterion) {
    #[cfg(feature = "nvidia-of")]
    {
        use remoteway_interpolate::backends::nvidia_optical_flow::NvidiaOpticalFlowInterpolator;

        let interp = match NvidiaOpticalFlowInterpolator::new() {
            Ok(i) => i,
            Err(e) => {
                eprintln!("nvidia-of: skipping — {e}");
                return;
            }
        };

        let mut group = c.benchmark_group("nvidia_optical_flow");
        for &(w, h, label) in RESOLUTIONS {
            let a = make_frame(w, h, 50, 0);
            let b = make_frame(w, h, 200, 16_666_667);
            group.bench_with_input(
                BenchmarkId::new("interpolate", label),
                &(a, b),
                |bench, (a, b)| {
                    bench.iter(|| interp.interpolate(a, b, 0.5).unwrap());
                },
            );
        }
        group.finish();
    }
    #[cfg(not(feature = "nvidia-of"))]
    {
    }
}

// ---------------------------------------------------------------------------
// Comparison: all backends side-by-side at 1080p
// ---------------------------------------------------------------------------

fn bench_comparison_1080p(c: &mut Criterion) {
    let a = make_frame(1920, 1080, 50, 0);
    let b = make_frame(1920, 1080, 200, 16_666_667);

    let mut group = c.benchmark_group("comparison_1080p");

    {
        let interp = LinearBlendInterpolator;
        group.bench_function("linear_blend", |bench| {
            bench.iter(|| interp.interpolate(&a, &b, 0.5).unwrap());
        });
    }

    #[cfg(feature = "wgpu-backend")]
    {
        use remoteway_interpolate::backends::wgpu_optical_flow::WgpuOpticalFlowInterpolator;
        if let Ok(interp) = WgpuOpticalFlowInterpolator::new() {
            group.bench_function("wgpu_optical_flow", |bench| {
                bench.iter(|| interp.interpolate(&a, &b, 0.5).unwrap());
            });
        }
    }

    #[cfg(feature = "fsr2")]
    {
        use remoteway_interpolate::backends::fsr2::Fsr2Interpolator;
        if let Ok(interp) = Fsr2Interpolator::new() {
            group.bench_function("fsr2", |bench| {
                bench.iter(|| interp.interpolate(&a, &b, 0.5).unwrap());
            });
        }
    }

    #[cfg(feature = "fsr3")]
    {
        use remoteway_interpolate::backends::fsr3::Fsr3Interpolator;
        if let Ok(interp) = Fsr3Interpolator::new() {
            group.bench_function("fsr3", |bench| {
                bench.iter(|| interp.interpolate(&a, &b, 0.5).unwrap());
            });
        }
    }

    #[cfg(feature = "nvidia-of")]
    {
        use remoteway_interpolate::backends::nvidia_optical_flow::NvidiaOpticalFlowInterpolator;
        if let Ok(interp) = NvidiaOpticalFlowInterpolator::new() {
            group.bench_function("nvidia_optical_flow", |bench| {
                bench.iter(|| interp.interpolate(&a, &b, 0.5).unwrap());
            });
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Comparison: all backends side-by-side at 4K
// ---------------------------------------------------------------------------

fn bench_comparison_4k(c: &mut Criterion) {
    let a = make_frame(3840, 2160, 50, 0);
    let b = make_frame(3840, 2160, 200, 16_666_667);

    let mut group = c.benchmark_group("comparison_4K");

    {
        let interp = LinearBlendInterpolator;
        group.bench_function("linear_blend", |bench| {
            bench.iter(|| interp.interpolate(&a, &b, 0.5).unwrap());
        });
    }

    #[cfg(feature = "wgpu-backend")]
    {
        use remoteway_interpolate::backends::wgpu_optical_flow::WgpuOpticalFlowInterpolator;
        if let Ok(interp) = WgpuOpticalFlowInterpolator::new() {
            group.bench_function("wgpu_optical_flow", |bench| {
                bench.iter(|| interp.interpolate(&a, &b, 0.5).unwrap());
            });
        }
    }

    #[cfg(feature = "fsr2")]
    {
        use remoteway_interpolate::backends::fsr2::Fsr2Interpolator;
        if let Ok(interp) = Fsr2Interpolator::new() {
            group.bench_function("fsr2", |bench| {
                bench.iter(|| interp.interpolate(&a, &b, 0.5).unwrap());
            });
        }
    }

    #[cfg(feature = "fsr3")]
    {
        use remoteway_interpolate::backends::fsr3::Fsr3Interpolator;
        if let Ok(interp) = Fsr3Interpolator::new() {
            group.bench_function("fsr3", |bench| {
                bench.iter(|| interp.interpolate(&a, &b, 0.5).unwrap());
            });
        }
    }

    #[cfg(feature = "nvidia-of")]
    {
        use remoteway_interpolate::backends::nvidia_optical_flow::NvidiaOpticalFlowInterpolator;
        if let Ok(interp) = NvidiaOpticalFlowInterpolator::new() {
            group.bench_function("nvidia_optical_flow", |bench| {
                bench.iter(|| interp.interpolate(&a, &b, 0.5).unwrap());
            });
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_linear_blend,
    bench_linear_blend_latency,
    bench_wgpu_optical_flow,
    bench_fsr2,
    bench_fsr3,
    bench_nvidia_of,
    bench_comparison_1080p,
    bench_comparison_4k,
);
criterion_main!(benches);

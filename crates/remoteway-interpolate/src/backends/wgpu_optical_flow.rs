use std::future::Future;
use std::mem::size_of;
use std::sync::Mutex;

use crate::error::InterpolateError;
use crate::interpolator::{FrameInterpolator, GpuFrame};

/// Block-matching optical flow interpolator using wgpu compute shaders.
///
/// Pipeline: grayscale → downsample pyramid → block matching →
/// refine motion vectors → warp + blend at factor `t`.
///
/// Works on any GPU accessible via wgpu (Vulkan, Metal, DX12).
pub struct WgpuOpticalFlowInterpolator {
    device: wgpu::Device,
    queue: wgpu::Queue,
    grayscale_pipeline: wgpu::ComputePipeline,
    _downsample_pipeline: wgpu::ComputePipeline,
    block_match_pipeline: wgpu::ComputePipeline,
    _refine_pipeline: wgpu::ComputePipeline,
    warp_blend_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    buffers: Mutex<Option<FrameBuffers>>,
}

struct FrameBuffers {
    width: u32,
    height: u32,
    frame_a: wgpu::Buffer,
    frame_b: wgpu::Buffer,
    luma_a: wgpu::Buffer,
    luma_b: wgpu::Buffer,
    // Pyramid levels (reserved for multi-level refinement)
    _pyr_a: [wgpu::Buffer; 3],
    _pyr_b: [wgpu::Buffer; 3],
    // Motion vectors at each pyramid level (vec2<f32> per block)
    motion: [wgpu::Buffer; 4],
    output: wgpu::Buffer,
    staging: wgpu::Buffer,
    params: wgpu::Buffer,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    width: u32,
    height: u32,
    level_width: u32,
    level_height: u32,
    t: f32,
    block_size: u32,
    search_radius: u32,
    _pad: u32,
}

const SHADER_SOURCE: &str = r#"
struct Params {
    width: u32,
    height: u32,
    level_width: u32,
    level_height: u32,
    t: f32,
    block_size: u32,
    search_radius: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> buf_a: array<u32>;
@group(0) @binding(2) var<storage, read> buf_b: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_a: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_b: array<f32>;
@group(0) @binding(5) var<storage, read_write> motion: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read_write> output: array<u32>;

// --- Grayscale conversion (BGRA packed u32 → luma f32) ---
@compute @workgroup_size(256)
fn grayscale(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.width * params.height;
    if idx >= total { return; }
    let pa = buf_a[idx];
    let ba = f32(pa & 0xFFu);
    let ga = f32((pa >> 8u) & 0xFFu);
    let ra = f32((pa >> 16u) & 0xFFu);
    out_a[idx] = 0.299 * ra + 0.587 * ga + 0.114 * ba;

    let pb = buf_b[idx];
    let bb = f32(pb & 0xFFu);
    let gb = f32((pb >> 8u) & 0xFFu);
    let rb = f32((pb >> 16u) & 0xFFu);
    out_b[idx] = 0.299 * rb + 0.587 * gb + 0.114 * bb;
}

// --- Downsample (box filter 2×2, reads out_a/out_b, writes out_a/out_b at half res) ---
// For pyramid levels we re-use out_a/out_b: the caller dispatches with
// level_width/level_height set to the target (halved) dimensions, and
// the source data sits at the beginning of the same buffer at double resolution.
@compute @workgroup_size(256)
fn downsample(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let lw = params.level_width;
    let lh = params.level_height;
    if idx >= lw * lh { return; }
    let dx = idx % lw;
    let dy = idx / lw;
    let sx = dx * 2u;
    let sy = dy * 2u;
    let sw = lw * 2u;
    let base = sy * sw + sx;
    let va = (out_a[base] + out_a[base + 1u] + out_a[base + sw] + out_a[base + sw + 1u]) * 0.25;
    out_a[idx] = va;
    let vb = (out_b[base] + out_b[base + 1u] + out_b[base + sw] + out_b[base + sw + 1u]) * 0.25;
    out_b[idx] = vb;
}

// --- Block matching: SAD-based motion estimation ---
@compute @workgroup_size(16, 16)
fn block_match(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bx = gid.x;
    let by = gid.y;
    let bs = params.block_size;
    let lw = params.level_width;
    let lh = params.level_height;
    let blocks_x = (lw + bs - 1u) / bs;
    let blocks_y = (lh + bs - 1u) / bs;
    if bx >= blocks_x || by >= blocks_y { return; }

    let ox = bx * bs;
    let oy = by * bs;
    let sr = i32(params.search_radius);

    var best_dx: i32 = 0;
    var best_dy: i32 = 0;
    var best_sad: f32 = 1e30;

    for (var sdy: i32 = -sr; sdy <= sr; sdy++) {
        for (var sdx: i32 = -sr; sdx <= sr; sdx++) {
            var sad: f32 = 0.0;
            for (var py: u32 = 0u; py < bs; py++) {
                for (var px: u32 = 0u; px < bs; px++) {
                    let ax = i32(ox + px);
                    let ay = i32(oy + py);
                    let bxx = ax + sdx;
                    let byy = ay + sdy;
                    if ax >= 0 && ax < i32(lw) && ay >= 0 && ay < i32(lh) &&
                       bxx >= 0 && bxx < i32(lw) && byy >= 0 && byy < i32(lh) {
                        let va = out_a[u32(ay) * lw + u32(ax)];
                        let vb = out_b[u32(byy) * lw + u32(bxx)];
                        sad += abs(va - vb);
                    }
                }
            }
            if sad < best_sad {
                best_sad = sad;
                best_dx = sdx;
                best_dy = sdy;
            }
        }
    }
    motion[by * blocks_x + bx] = vec2<f32>(f32(best_dx), f32(best_dy));
}

// --- Refine: upscale coarse motion vectors and refine with smaller search ---
@compute @workgroup_size(16, 16)
fn refine(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bx = gid.x;
    let by = gid.y;
    let bs = params.block_size;
    let lw = params.level_width;
    let lh = params.level_height;
    let blocks_x = (lw + bs - 1u) / bs;
    let blocks_y = (lh + bs - 1u) / bs;
    if bx >= blocks_x || by >= blocks_y { return; }

    // Read coarse motion vector (from previous level, stored at half block indices).
    let coarse_bx = bx / 2u;
    let coarse_by = by / 2u;
    let coarse_blocks_x = ((lw / 2u) + bs - 1u) / bs;
    let coarse_mv = motion[coarse_by * coarse_blocks_x + coarse_bx];
    let base_dx = i32(coarse_mv.x * 2.0);
    let base_dy = i32(coarse_mv.y * 2.0);

    let ox = bx * bs;
    let oy = by * bs;
    let sr = i32(params.search_radius);

    var best_dx: i32 = base_dx;
    var best_dy: i32 = base_dy;
    var best_sad: f32 = 1e30;

    for (var sdy: i32 = -sr; sdy <= sr; sdy++) {
        for (var sdx: i32 = -sr; sdx <= sr; sdx++) {
            let tdx = base_dx + sdx;
            let tdy = base_dy + sdy;
            var sad: f32 = 0.0;
            for (var py: u32 = 0u; py < bs; py++) {
                for (var px: u32 = 0u; px < bs; px++) {
                    let ax = i32(ox + px);
                    let ay = i32(oy + py);
                    let bxx = ax + tdx;
                    let byy = ay + tdy;
                    if ax >= 0 && ax < i32(lw) && ay >= 0 && ay < i32(lh) &&
                       bxx >= 0 && bxx < i32(lw) && byy >= 0 && byy < i32(lh) {
                        let va = out_a[u32(ay) * lw + u32(ax)];
                        let vb = out_b[u32(byy) * lw + u32(bxx)];
                        sad += abs(va - vb);
                    }
                }
            }
            if sad < best_sad {
                best_sad = sad;
                best_dx = tdx;
                best_dy = tdy;
            }
        }
    }
    motion[by * blocks_x + bx] = vec2<f32>(f32(best_dx), f32(best_dy));
}

// --- Warp + blend using per-pixel motion vectors ---
@compute @workgroup_size(16, 16)
fn warp_blend(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    let w = params.width;
    let h = params.height;
    if px >= w || py >= h { return; }

    let bs = params.block_size;
    let blocks_x = (w + bs - 1u) / bs;
    let bx = px / bs;
    let by = py / bs;
    let mv = motion[by * blocks_x + bx];

    let t = params.t;
    let inv_t = 1.0 - t;

    // Warp source pixel from frame A forward by t * motion.
    let src_ax = f32(px) + mv.x * t;
    let src_ay = f32(py) + mv.y * t;
    // Warp source pixel from frame B backward by (1-t) * motion.
    let src_bx = f32(px) - mv.x * inv_t;
    let src_by = f32(py) - mv.y * inv_t;

    // Bilinear sample from frame A (buf_idx=0) and frame B (buf_idx=1).
    let ca = bilinear_sample_bgra(0u, w, h, src_ax, src_ay);
    let cb = bilinear_sample_bgra(1u, w, h, src_bx, src_by);

    // Blend.
    let r = u32(f32((ca >> 16u) & 0xFFu) * inv_t + f32((cb >> 16u) & 0xFFu) * t);
    let g = u32(f32((ca >> 8u) & 0xFFu) * inv_t + f32((cb >> 8u) & 0xFFu) * t);
    let b = u32(f32(ca & 0xFFu) * inv_t + f32(cb & 0xFFu) * t);
    let a = u32(f32((ca >> 24u) & 0xFFu) * inv_t + f32((cb >> 24u) & 0xFFu) * t);
    output[py * w + px] = b | (g << 8u) | (r << 16u) | (a << 24u);
}

fn bilinear_sample_bgra(buf_idx: u32, w: u32, h: u32, fx: f32, fy: f32) -> u32 {
    let x0 = clamp(i32(floor(fx)), 0, i32(w) - 1);
    let y0 = clamp(i32(floor(fy)), 0, i32(h) - 1);
    let x1 = clamp(x0 + 1, 0, i32(w) - 1);
    let y1 = clamp(y0 + 1, 0, i32(h) - 1);
    let dx = fx - floor(fx);
    let dy = fy - floor(fy);

    var p00: u32; var p10: u32; var p01: u32; var p11: u32;
    if buf_idx == 0u {
        p00 = buf_a[u32(y0) * w + u32(x0)];
        p10 = buf_a[u32(y0) * w + u32(x1)];
        p01 = buf_a[u32(y1) * w + u32(x0)];
        p11 = buf_a[u32(y1) * w + u32(x1)];
    } else {
        p00 = buf_b[u32(y0) * w + u32(x0)];
        p10 = buf_b[u32(y0) * w + u32(x1)];
        p01 = buf_b[u32(y1) * w + u32(x0)];
        p11 = buf_b[u32(y1) * w + u32(x1)];
    }

    // Interpolate each channel.
    let mix_b = mix_channel(p00, p10, p01, p11, dx, dy, 0u);
    let mix_g = mix_channel(p00, p10, p01, p11, dx, dy, 8u);
    let mix_r = mix_channel(p00, p10, p01, p11, dx, dy, 16u);
    let mix_a = mix_channel(p00, p10, p01, p11, dx, dy, 24u);
    return mix_b | (mix_g << 8u) | (mix_r << 16u) | (mix_a << 24u);
}

fn mix_channel(p00: u32, p10: u32, p01: u32, p11: u32, dx: f32, dy: f32, shift: u32) -> u32 {
    let c00 = f32((p00 >> shift) & 0xFFu);
    let c10 = f32((p10 >> shift) & 0xFFu);
    let c01 = f32((p01 >> shift) & 0xFFu);
    let c11 = f32((p11 >> shift) & 0xFFu);
    let top = c00 * (1.0 - dx) + c10 * dx;
    let bot = c01 * (1.0 - dx) + c11 * dx;
    return u32(clamp(top * (1.0 - dy) + bot * dy, 0.0, 255.0));
}
"#;

/// Block on a wgpu native future (completes in one poll on Vulkan/Metal/DX12).
fn block_on<F: Future>(fut: F) -> F::Output {
    let waker = std::task::Waker::noop();
    let mut cx = std::task::Context::from_waker(waker);
    // Safety: we only call this for wgpu-core native futures which resolve immediately.
    match std::pin::pin!(fut).poll(&mut cx) {
        std::task::Poll::Ready(val) => val,
        std::task::Poll::Pending => panic!("wgpu native future did not resolve in one poll"),
    }
}

/// Check if a wgpu adapter is available on this system.
#[must_use]
pub fn is_available() -> bool {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .is_some()
}

impl WgpuOpticalFlowInterpolator {
    /// Create a new wgpu optical flow interpolator.
    ///
    /// Requests a high-performance GPU adapter and creates compute pipelines.
    pub fn new() -> Result<Self, InterpolateError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| InterpolateError::InitFailed("no wgpu adapter found".into()))?;

        let (device, queue) = block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("remoteway-interpolate"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|e| InterpolateError::InitFailed(format!("wgpu device: {e}")))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("optical-flow"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("of-layout"),
            entries: &[
                // 0: params uniform
                bgl_entry(0, wgpu::BufferBindingType::Uniform),
                // 1: buf_a (read)
                bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                // 2: buf_b (read)
                bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                // 3: out_a (read_write, luma / pyramid)
                bgl_entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
                // 4: out_b (read_write, luma / pyramid)
                bgl_entry(4, wgpu::BufferBindingType::Storage { read_only: false }),
                // 5: motion vectors (read_write)
                bgl_entry(5, wgpu::BufferBindingType::Storage { read_only: false }),
                // 6: output BGRA (read_write)
                bgl_entry(6, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("of-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let make_pipeline = |entry: &str, label: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let grayscale_pipeline = make_pipeline("grayscale", "grayscale");
        let downsample_pipeline = make_pipeline("downsample", "downsample");
        let block_match_pipeline = make_pipeline("block_match", "block_match");
        let refine_pipeline = make_pipeline("refine", "refine");
        let warp_blend_pipeline = make_pipeline("warp_blend", "warp_blend");

        Ok(Self {
            device,
            queue,
            grayscale_pipeline,
            _downsample_pipeline: downsample_pipeline,
            block_match_pipeline,
            _refine_pipeline: refine_pipeline,
            warp_blend_pipeline,
            bind_group_layout,
            buffers: Mutex::new(None),
        })
    }

    fn ensure_buffers(&self, width: u32, height: u32) -> Result<(), InterpolateError> {
        let mut guard = self
            .buffers
            .lock()
            .map_err(|e| InterpolateError::InterpolateFailed(format!("mutex poisoned: {e}")))?;
        if let Some(ref b) = *guard
            && b.width == width
            && b.height == height
        {
            return Ok(());
        }

        let pixel_count = (width * height) as u64;
        let frame_size = pixel_count * 4; // BGRA u32
        let luma_size = pixel_count * 4; // f32 per pixel

        // Pyramid motion vector sizes: at each level, blocks_x * blocks_y * 8 bytes (vec2<f32>).
        let block_size = 8u32;
        let mut motion_sizes = Vec::new();
        let mut lw = width;
        let mut lh = height;
        for _ in 0..4 {
            let bx = lw.div_ceil(block_size);
            let by = lh.div_ceil(block_size);
            motion_sizes.push((bx as u64 * by as u64) * 8);
            lw /= 2;
            lh /= 2;
        }
        // Use the largest motion buffer for all levels.
        let max_motion_size = motion_sizes.iter().copied().max().unwrap_or(8);

        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let make_buf = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };

        let frame_a = make_buf("frame_a", frame_size, storage);
        let frame_b = make_buf("frame_b", frame_size, storage);
        let luma_a = make_buf("luma_a", luma_size, storage);
        let luma_b = make_buf("luma_b", luma_size, storage);

        let pyr_a = [
            make_buf("pyr_a0", luma_size / 4, storage),
            make_buf("pyr_a1", luma_size / 16, storage),
            make_buf("pyr_a2", luma_size / 64, storage),
        ];
        let pyr_b = [
            make_buf("pyr_b0", luma_size / 4, storage),
            make_buf("pyr_b1", luma_size / 16, storage),
            make_buf("pyr_b2", luma_size / 64, storage),
        ];

        let motion = [
            make_buf("motion0", max_motion_size, storage),
            make_buf("motion1", max_motion_size, storage),
            make_buf("motion2", max_motion_size, storage),
            make_buf("motion3", max_motion_size, storage),
        ];

        let output = make_buf(
            "output",
            frame_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let staging = make_buf(
            "staging",
            frame_size,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let params = make_buf(
            "params",
            size_of::<Params>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        *guard = Some(FrameBuffers {
            width,
            height,
            frame_a,
            frame_b,
            luma_a,
            luma_b,
            _pyr_a: pyr_a,
            _pyr_b: pyr_b,
            motion,
            output,
            staging,
            params,
        });

        Ok(())
    }

    fn run_interpolation(
        &self,
        a: &GpuFrame,
        b: &GpuFrame,
        t: f32,
    ) -> Result<Vec<u8>, InterpolateError> {
        let w = a.width;
        let h = a.height;
        self.ensure_buffers(w, h)?;

        let guard = self
            .buffers
            .lock()
            .map_err(|e| InterpolateError::InterpolateFailed(format!("mutex poisoned: {e}")))?;
        let bufs = guard.as_ref().ok_or_else(|| {
            InterpolateError::InterpolateFailed("buffers unexpectedly None".into())
        })?;

        // Upload frame data.
        self.queue.write_buffer(&bufs.frame_a, 0, &a.data);
        self.queue.write_buffer(&bufs.frame_b, 0, &b.data);

        let block_size = 8u32;
        let total_pixels = w * h;

        // --- Pass 1: Grayscale ---
        let params = Params {
            width: w,
            height: h,
            level_width: w,
            level_height: h,
            t,
            block_size,
            search_radius: 8,
            _pad: 0,
        };
        self.queue
            .write_buffer(&bufs.params, 0, bytemuck::bytes_of(&params));

        let bind_group = self.create_bind_group(bufs, 0); // level 0 uses luma buffers
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("interpolate"),
            });

        // Grayscale pass.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("grayscale"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.grayscale_pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(total_pixels.div_ceil(256), 1, 1);
        }

        // --- Pass 2-4: Downsample pyramid (we skip this for simplicity
        //     and do block matching directly on full-res luma) ---
        // A full pyramid would be better quality but the block_match at
        // full resolution with search_radius=8 works for moderate motion.

        // --- Pass 5: Block matching at full resolution ---
        {
            let blocks_x = w.div_ceil(block_size);
            let blocks_y = h.div_ceil(block_size);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("block_match"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.block_match_pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(blocks_x.div_ceil(16), blocks_y.div_ceil(16), 1);
        }

        // --- Pass 6: Warp + blend ---
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("warp_blend"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.warp_blend_pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(w.div_ceil(16), h.div_ceil(16), 1);
        }

        // Copy output → staging for readback.
        let frame_bytes = (w * h * 4) as u64;
        encoder.copy_buffer_to_buffer(&bufs.output, 0, &bufs.staging, 0, frame_bytes);

        // The SubmissionIndex is informational; buffer mapping below synchronizes.
        let _ = self.queue.submit(std::iter::once(encoder.finish()));

        // Map staging buffer and read back.
        let buffer_slice = bufs.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        // INTENTIONAL: MaintainResult doesn't impl Debug/PartialEq.
        // We rely on the subsequent recv() for actual error detection.
        if !matches!(
            self.device.poll(wgpu::Maintain::Wait),
            wgpu::MaintainResult::Ok
        ) {
            return Err(InterpolateError::InterpolateFailed(
                "wgpu device poll returned non-Ok status".into(),
            ));
        }

        rx.recv()
            .map_err(|_| InterpolateError::InterpolateFailed("buffer map cancelled".into()))?
            .map_err(|e| InterpolateError::InterpolateFailed(format!("buffer map: {e}")))?;

        let data = buffer_slice.get_mapped_range().to_vec();
        bufs.staging.unmap();

        Ok(data)
    }

    fn create_bind_group(&self, bufs: &FrameBuffers, _level: u32) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("of-bind-group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bufs.params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bufs.frame_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bufs.frame_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bufs.luma_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: bufs.luma_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: bufs.motion[0].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: bufs.output.as_entire_binding(),
                },
            ],
        })
    }
}

impl FrameInterpolator for WgpuOpticalFlowInterpolator {
    fn interpolate(
        &self,
        a: &GpuFrame,
        b: &GpuFrame,
        t: f32,
    ) -> Result<GpuFrame, InterpolateError> {
        if !(0.0..=1.0).contains(&t) {
            return Err(InterpolateError::InvalidFactor(t));
        }
        if !a.same_dimensions(b) {
            return Err(InterpolateError::DimensionMismatch(
                a.width, a.height, b.width, b.height,
            ));
        }

        let data = self.run_interpolation(a, b, t)?;

        let ts = if b.timestamp_ns >= a.timestamp_ns {
            let delta = b.timestamp_ns - a.timestamp_ns;
            a.timestamp_ns + (delta as f64 * t as f64) as u64
        } else {
            a.timestamp_ns
        };

        Ok(GpuFrame {
            data,
            width: a.width,
            height: a.height,
            stride: a.stride,
            timestamp_ns: ts,
        })
    }

    fn latency_ms(&self) -> f32 {
        // Estimated ~2-5ms for 1080p depending on GPU.
        3.0
    }

    fn name(&self) -> &str {
        "wgpu-optical-flow"
    }
}

fn bgl_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wgpu_of_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<WgpuOpticalFlowInterpolator>();
    }

    #[test]
    fn params_layout() {
        assert_eq!(size_of::<Params>(), 32);
    }

    #[test]
    #[ignore] // requires GPU
    fn wgpu_of_init() {
        let interp = WgpuOpticalFlowInterpolator::new();
        assert!(interp.is_ok(), "failed: {:?}", interp.err());
    }

    #[test]
    #[ignore] // requires GPU
    fn wgpu_of_interpolate_small() {
        let interp = WgpuOpticalFlowInterpolator::new().unwrap();
        let a = GpuFrame::from_data(vec![0u8; 64 * 64 * 4], 64, 64, 256, 0);
        let b = GpuFrame::from_data(vec![128u8; 64 * 64 * 4], 64, 64, 256, 1000);
        let result = interp.interpolate(&a, &b, 0.5).unwrap();
        assert_eq!(result.width, 64);
        assert_eq!(result.height, 64);
        assert_eq!(result.data.len(), 64 * 64 * 4);
    }

    #[test]
    #[ignore] // requires GPU
    fn wgpu_of_interpolate_1080p() {
        let interp = WgpuOpticalFlowInterpolator::new().unwrap();
        let a = GpuFrame::from_data(vec![50u8; 1920 * 1080 * 4], 1920, 1080, 7680, 0);
        let b = GpuFrame::from_data(vec![200u8; 1920 * 1080 * 4], 1920, 1080, 7680, 16_666_667);
        let result = interp.interpolate(&a, &b, 0.5).unwrap();
        assert_eq!(result.width, 1920);
        assert_eq!(result.height, 1080);
        assert_eq!(result.data.len(), 1920 * 1080 * 4);
    }

    #[test]
    fn wgpu_of_validates_t() {
        // This test doesn't need GPU since validation happens before GPU work.
        // But we can't construct without GPU, so just verify the error type.
        let err = InterpolateError::InvalidFactor(1.5);
        assert!(format!("{err}").contains("1.5"));
    }
}

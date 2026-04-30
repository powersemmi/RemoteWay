use std::sync::{Arc, Mutex};

use ash::vk;

use crate::error::InterpolateError;
use crate::interpolator::{FrameInterpolator, GpuFrame};

use super::vulkan_context::VulkanContext;

const MOTION_EST_SPV: &[u8] = include_bytes!("../shaders/motion_est.spv");
const WARP_BLEND_SPV: &[u8] = include_bytes!("../shaders/warp_blend.spv");

/// Push constants shared between motion estimation and warp/blend shaders.
#[repr(C)]
#[derive(Clone, Copy)]
struct PushConstants {
    width: u32,
    height: u32,
    t: f32,
    block_size: u32,
    search_radius: u32,
}

/// Vulkan compute-based motion-compensated frame interpolation.
///
/// Approximates AMD FSR 2 approach: block-matching motion estimation
/// followed by motion-compensated warp and blend. Uses SPIR-V compute
/// shaders running on any Vulkan 1.1+ GPU.
pub struct Fsr2Interpolator {
    ctx: Arc<Mutex<VulkanContext>>,
    motion_pipeline: vk::Pipeline,
    warp_pipeline: vk::Pipeline,
    motion_pipeline_layout: vk::PipelineLayout,
    warp_pipeline_layout: vk::PipelineLayout,
    motion_desc_layout: vk::DescriptorSetLayout,
    warp_desc_layout: vk::DescriptorSetLayout,
    desc_pool: vk::DescriptorPool,
    cached: Mutex<Option<Fsr2Resources>>,
    block_size: u32,
    search_radius: u32,
}

struct Fsr2Resources {
    width: u32,
    height: u32,
    frame_a_buf: vk::Buffer,
    frame_a_mem: vk::DeviceMemory,
    frame_b_buf: vk::Buffer,
    frame_b_mem: vk::DeviceMemory,
    motion_buf: vk::Buffer,
    motion_mem: vk::DeviceMemory,
    output_buf: vk::Buffer,
    output_mem: vk::DeviceMemory,
    staging_a_buf: vk::Buffer,
    staging_a_mem: vk::DeviceMemory,
    staging_b_buf: vk::Buffer,
    staging_b_mem: vk::DeviceMemory,
    readback_buf: vk::Buffer,
    readback_mem: vk::DeviceMemory,
    motion_desc_set: vk::DescriptorSet,
    warp_desc_set: vk::DescriptorSet,
    cmd: vk::CommandBuffer,
}

impl Fsr2Interpolator {
    /// Create a new FSR2-style Vulkan compute interpolator.
    pub fn new() -> Result<Self, InterpolateError> {
        Self::with_params(8, 8)
    }

    /// Create with custom block size and search radius.
    pub(crate) fn with_params(
        block_size: u32,
        search_radius: u32,
    ) -> Result<Self, InterpolateError> {
        let ctx = Arc::new(Mutex::new(VulkanContext::new(&[])?));
        let guard = ctx.lock().unwrap();

        // Create shader modules.
        let motion_spv = ash::util::read_spv(&mut std::io::Cursor::new(MOTION_EST_SPV))
            .map_err(|e| InterpolateError::InitFailed(format!("motion SPIR-V: {e}")))?;
        let warp_spv = ash::util::read_spv(&mut std::io::Cursor::new(WARP_BLEND_SPV))
            .map_err(|e| InterpolateError::InitFailed(format!("warp SPIR-V: {e}")))?;

        let motion_module_info = vk::ShaderModuleCreateInfo::default().code(&motion_spv);
        let warp_module_info = vk::ShaderModuleCreateInfo::default().code(&warp_spv);

        let motion_module = unsafe { guard.device.create_shader_module(&motion_module_info, None) }
            .map_err(|e| InterpolateError::InitFailed(format!("motion shader: {e}")))?;
        let warp_module = unsafe { guard.device.create_shader_module(&warp_module_info, None) }
            .map_err(|e| InterpolateError::InitFailed(format!("warp shader: {e}")))?;

        // Descriptor set layouts.
        // Motion estimation: frame_a(0), frame_b(1), motion(2)
        let motion_bindings = [
            desc_binding(0, vk::DescriptorType::STORAGE_BUFFER),
            desc_binding(1, vk::DescriptorType::STORAGE_BUFFER),
            desc_binding(2, vk::DescriptorType::STORAGE_BUFFER),
        ];
        let motion_desc_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&motion_bindings);
        let motion_desc_layout = unsafe {
            guard
                .device
                .create_descriptor_set_layout(&motion_desc_layout_info, None)
        }
        .map_err(|e| InterpolateError::InitFailed(format!("desc layout: {e}")))?;

        // Warp blend: frame_a(0), frame_b(1), motion(2), output(3)
        let warp_bindings = [
            desc_binding(0, vk::DescriptorType::STORAGE_BUFFER),
            desc_binding(1, vk::DescriptorType::STORAGE_BUFFER),
            desc_binding(2, vk::DescriptorType::STORAGE_BUFFER),
            desc_binding(3, vk::DescriptorType::STORAGE_BUFFER),
        ];
        let warp_desc_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&warp_bindings);
        let warp_desc_layout = unsafe {
            guard
                .device
                .create_descriptor_set_layout(&warp_desc_layout_info, None)
        }
        .map_err(|e| InterpolateError::InitFailed(format!("desc layout: {e}")))?;

        // Push constant range.
        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<PushConstants>() as u32);

        // Pipeline layouts.
        let motion_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&motion_desc_layout))
            .push_constant_ranges(std::slice::from_ref(&push_range));
        let motion_pipeline_layout = unsafe {
            guard
                .device
                .create_pipeline_layout(&motion_layout_info, None)
        }
        .map_err(|e| InterpolateError::InitFailed(format!("pipeline layout: {e}")))?;

        let warp_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&warp_desc_layout))
            .push_constant_ranges(std::slice::from_ref(&push_range));
        let warp_pipeline_layout =
            unsafe { guard.device.create_pipeline_layout(&warp_layout_info, None) }
                .map_err(|e| InterpolateError::InitFailed(format!("pipeline layout: {e}")))?;

        // Compute pipelines.
        let motion_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(motion_module)
            .name(c"main");
        let motion_pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(motion_stage)
            .layout(motion_pipeline_layout);

        let warp_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(warp_module)
            .name(c"main");
        let warp_pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(warp_stage)
            .layout(warp_pipeline_layout);

        let pipelines = unsafe {
            guard.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[motion_pipeline_info, warp_pipeline_info],
                None,
            )
        }
        .map_err(|(_pipelines, e)| InterpolateError::InitFailed(format!("pipelines: {e}")))?;

        let motion_pipeline = pipelines[0];
        let warp_pipeline = pipelines[1];

        // Descriptor pool.
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 16,
        }];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(4)
            .pool_sizes(&pool_sizes);
        let desc_pool = unsafe { guard.device.create_descriptor_pool(&pool_info, None) }
            .map_err(|e| InterpolateError::InitFailed(format!("desc pool: {e}")))?;

        // Cleanup shader modules (no longer needed after pipeline creation).
        unsafe {
            guard.device.destroy_shader_module(motion_module, None);
            guard.device.destroy_shader_module(warp_module, None);
        }

        drop(guard);

        Ok(Self {
            ctx,
            motion_pipeline,
            warp_pipeline,
            motion_pipeline_layout,
            warp_pipeline_layout,
            motion_desc_layout,
            warp_desc_layout,
            desc_pool,
            cached: Mutex::new(None),
            block_size,
            search_radius,
        })
    }

    fn ensure_resources(&self, width: u32, height: u32) -> Result<(), InterpolateError> {
        let mut cache = self.cached.lock().unwrap();
        if let Some(ref r) = *cache {
            if r.width == width && r.height == height {
                return Ok(());
            }
            // Clean up old resources and reset descriptor pool.
            let guard = self.ctx.lock().unwrap();
            destroy_resources(&guard.device, cache.take().unwrap());
            unsafe {
                guard
                    .device
                    .reset_descriptor_pool(self.desc_pool, vk::DescriptorPoolResetFlags::empty())
                    .ok();
            }
            drop(guard);
        }

        let guard = self.ctx.lock().unwrap();

        let frame_size = (width * height * 4) as u64;
        let blocks_x = width.div_ceil(self.block_size);
        let blocks_y = height.div_ceil(self.block_size);
        let motion_size = (blocks_x * blocks_y * 8) as u64; // vec2<f32>

        let host_visible =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let device_local = vk::MemoryPropertyFlags::DEVICE_LOCAL;

        // Try device-local, fall back to host-visible.
        let try_create = |size: u64,
                          usage: vk::BufferUsageFlags|
         -> Result<(vk::Buffer, vk::DeviceMemory), InterpolateError> {
            guard
                .create_buffer(size, usage, device_local)
                .or_else(|_| guard.create_buffer(size, usage, host_visible))
        };

        let storage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        let (frame_a_buf, frame_a_mem) = try_create(frame_size, storage)?;
        let (frame_b_buf, frame_b_mem) = try_create(frame_size, storage)?;
        let (motion_buf, motion_mem) =
            try_create(motion_size, vk::BufferUsageFlags::STORAGE_BUFFER)?;
        let (output_buf, output_mem) = try_create(
            frame_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
        )?;
        let staging_usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        let (staging_a_buf, staging_a_mem) =
            guard.create_buffer(frame_size, staging_usage, host_visible)?;
        let (staging_b_buf, staging_b_mem) =
            guard.create_buffer(frame_size, staging_usage, host_visible)?;
        let (readback_buf, readback_mem) =
            guard.create_buffer(frame_size, staging_usage, host_visible)?;

        // Allocate descriptor sets.
        let layouts = [self.motion_desc_layout, self.warp_desc_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.desc_pool)
            .set_layouts(&layouts);
        let desc_sets = unsafe { guard.device.allocate_descriptor_sets(&alloc_info) }
            .map_err(|e| InterpolateError::InterpolateFailed(format!("alloc desc: {e}")))?;

        // Write descriptor sets.
        let bufs_info: Vec<vk::DescriptorBufferInfo> = vec![
            // Motion set: frame_a(0), frame_b(1), motion(2)
            vk::DescriptorBufferInfo::default()
                .buffer(frame_a_buf)
                .range(frame_size),
            vk::DescriptorBufferInfo::default()
                .buffer(frame_b_buf)
                .range(frame_size),
            vk::DescriptorBufferInfo::default()
                .buffer(motion_buf)
                .range(motion_size),
            // Warp set: frame_a(0), frame_b(1), motion(2), output(3)
            vk::DescriptorBufferInfo::default()
                .buffer(frame_a_buf)
                .range(frame_size),
            vk::DescriptorBufferInfo::default()
                .buffer(frame_b_buf)
                .range(frame_size),
            vk::DescriptorBufferInfo::default()
                .buffer(motion_buf)
                .range(motion_size),
            vk::DescriptorBufferInfo::default()
                .buffer(output_buf)
                .range(frame_size),
        ];

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(desc_sets[0])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&bufs_info[0..1]),
            vk::WriteDescriptorSet::default()
                .dst_set(desc_sets[0])
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&bufs_info[1..2]),
            vk::WriteDescriptorSet::default()
                .dst_set(desc_sets[0])
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&bufs_info[2..3]),
            vk::WriteDescriptorSet::default()
                .dst_set(desc_sets[1])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&bufs_info[3..4]),
            vk::WriteDescriptorSet::default()
                .dst_set(desc_sets[1])
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&bufs_info[4..5]),
            vk::WriteDescriptorSet::default()
                .dst_set(desc_sets[1])
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&bufs_info[5..6]),
            vk::WriteDescriptorSet::default()
                .dst_set(desc_sets[1])
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&bufs_info[6..7]),
        ];

        unsafe { guard.device.update_descriptor_sets(&writes, &[]) };

        let cmd = guard.allocate_command_buffer()?;

        *cache = Some(Fsr2Resources {
            width,
            height,
            frame_a_buf,
            frame_a_mem,
            frame_b_buf,
            frame_b_mem,
            motion_buf,
            motion_mem,
            output_buf,
            output_mem,
            staging_a_buf,
            staging_a_mem,
            staging_b_buf,
            staging_b_mem,
            readback_buf,
            readback_mem,
            motion_desc_set: desc_sets[0],
            warp_desc_set: desc_sets[1],
            cmd,
        });

        Ok(())
    }

    fn run(&self, a: &GpuFrame, b: &GpuFrame, t: f32) -> Result<Vec<u8>, InterpolateError> {
        let w = a.width;
        let h = a.height;
        self.ensure_resources(w, h)?;

        let guard = self.ctx.lock().unwrap();
        let cache = self.cached.lock().unwrap();
        let res = cache.as_ref().unwrap();

        let frame_size = (w * h * 4) as u64;

        // Upload both frames to separate staging buffers (no GPU wait needed).
        guard.upload_to_buffer(res.staging_a_mem, &a.data)?;
        guard.upload_to_buffer(res.staging_b_mem, &b.data)?;

        let cmd = res.cmd;

        // Single command buffer: upload A + upload B + motion + warp + readback.
        unsafe {
            guard
                .device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .map_err(|e| InterpolateError::InterpolateFailed(format!("reset cmd: {e}")))?;
            guard
                .device
                .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
                .map_err(|e| InterpolateError::InterpolateFailed(format!("begin cmd: {e}")))?;

            // Copy staging_a → frame_a, staging_b → frame_b.
            let copy_region = vk::BufferCopy::default().size(frame_size);
            guard
                .device
                .cmd_copy_buffer(cmd, res.staging_a_buf, res.frame_a_buf, &[copy_region]);
            guard
                .device
                .cmd_copy_buffer(cmd, res.staging_b_buf, res.frame_b_buf, &[copy_region]);

            // Barrier: transfer → compute.
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            guard.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );

            let pc = PushConstants {
                width: w,
                height: h,
                t,
                block_size: self.block_size,
                search_radius: self.search_radius,
            };
            let pc_bytes = std::slice::from_raw_parts(
                &pc as *const PushConstants as *const u8,
                std::mem::size_of::<PushConstants>(),
            );

            let blocks_x = w.div_ceil(self.block_size);
            let blocks_y = h.div_ceil(self.block_size);

            // Pass 1: Motion estimation.
            guard.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.motion_pipeline,
            );
            guard.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.motion_pipeline_layout,
                0,
                &[res.motion_desc_set],
                &[],
            );
            guard.device.cmd_push_constants(
                cmd,
                self.motion_pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                pc_bytes,
            );
            guard
                .device
                .cmd_dispatch(cmd, blocks_x.div_ceil(16), blocks_y.div_ceil(16), 1);

            // Barrier: compute → compute.
            let comp_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            guard.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[comp_barrier],
                &[],
                &[],
            );

            // Pass 2: Warp + blend.
            guard
                .device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.warp_pipeline);
            guard.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.warp_pipeline_layout,
                0,
                &[res.warp_desc_set],
                &[],
            );
            guard.device.cmd_push_constants(
                cmd,
                self.warp_pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                pc_bytes,
            );
            guard
                .device
                .cmd_dispatch(cmd, w.div_ceil(16), h.div_ceil(16), 1);

            // Barrier: compute → transfer.
            let xfer_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            guard.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[xfer_barrier],
                &[],
                &[],
            );

            // Copy output → readback buffer.
            guard
                .device
                .cmd_copy_buffer(cmd, res.output_buf, res.readback_buf, &[copy_region]);

            guard
                .device
                .end_command_buffer(cmd)
                .map_err(|e| InterpolateError::InterpolateFailed(format!("end cmd: {e}")))?;
        }

        // Single submit + single fence wait.
        guard.submit_and_wait(cmd)?;

        // Readback from dedicated readback buffer.
        guard.read_from_buffer(res.readback_mem, frame_size as usize)
    }
}

impl FrameInterpolator for Fsr2Interpolator {
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

        let data = self.run(a, b, t)?;

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
        5.0
    }

    fn name(&self) -> &str {
        "fsr2"
    }
}

impl Drop for Fsr2Interpolator {
    fn drop(&mut self) {
        let guard = self.ctx.lock().unwrap();
        if let Some(res) = self.cached.lock().unwrap().take() {
            destroy_resources(&guard.device, res);
        }
        unsafe {
            guard.device.destroy_descriptor_pool(self.desc_pool, None);
            guard.device.destroy_pipeline(self.motion_pipeline, None);
            guard.device.destroy_pipeline(self.warp_pipeline, None);
            guard
                .device
                .destroy_pipeline_layout(self.motion_pipeline_layout, None);
            guard
                .device
                .destroy_pipeline_layout(self.warp_pipeline_layout, None);
            guard
                .device
                .destroy_descriptor_set_layout(self.motion_desc_layout, None);
            guard
                .device
                .destroy_descriptor_set_layout(self.warp_desc_layout, None);
        }
    }
}

fn destroy_resources(device: &ash::Device, res: Fsr2Resources) {
    unsafe {
        device.destroy_buffer(res.frame_a_buf, None);
        device.free_memory(res.frame_a_mem, None);
        device.destroy_buffer(res.frame_b_buf, None);
        device.free_memory(res.frame_b_mem, None);
        device.destroy_buffer(res.motion_buf, None);
        device.free_memory(res.motion_mem, None);
        device.destroy_buffer(res.output_buf, None);
        device.free_memory(res.output_mem, None);
        device.destroy_buffer(res.staging_a_buf, None);
        device.free_memory(res.staging_a_mem, None);
        device.destroy_buffer(res.staging_b_buf, None);
        device.free_memory(res.staging_b_mem, None);
        device.destroy_buffer(res.readback_buf, None);
        device.free_memory(res.readback_mem, None);
    }
}

fn desc_binding(binding: u32, ty: vk::DescriptorType) -> vk::DescriptorSetLayoutBinding<'static> {
    vk::DescriptorSetLayoutBinding::default()
        .binding(binding)
        .descriptor_type(ty)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fsr2_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Fsr2Interpolator>();
    }

    #[test]
    fn push_constants_size() {
        assert_eq!(std::mem::size_of::<PushConstants>(), 20);
    }

    #[test]
    #[ignore] // requires Vulkan
    fn fsr2_init() {
        let interp = Fsr2Interpolator::new();
        assert!(interp.is_ok(), "failed: {:?}", interp.err());
    }

    #[test]
    #[ignore] // requires Vulkan
    fn fsr2_interpolate_small() {
        let interp = Fsr2Interpolator::new().unwrap();
        let a = GpuFrame::from_data(vec![0u8; 64 * 64 * 4], 64, 64, 256, 0);
        let b = GpuFrame::from_data(vec![128u8; 64 * 64 * 4], 64, 64, 256, 1000);
        let result = interp.interpolate(&a, &b, 0.5).unwrap();
        assert_eq!(result.width, 64);
        assert_eq!(result.data.len(), 64 * 64 * 4);
    }
}

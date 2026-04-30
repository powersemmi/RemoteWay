use std::ffi::CStr;
use std::sync::{Arc, Mutex};

use ash::vk;

use crate::error::InterpolateError;
use crate::interpolator::{FrameInterpolator, GpuFrame};

use super::vulkan_context::VulkanContext;

const WARP_BLEND_SPV: &[u8] = include_bytes!("../shaders/warp_blend.spv");
const FLOW_CONVERT_SPV: &[u8] = include_bytes!("../shaders/flow_convert.spv");

/// Push constants for the warp/blend shader.
#[repr(C)]
#[derive(Clone, Copy)]
struct PushConstants {
    width: u32,
    height: u32,
    t: f32,
    block_size: u32,
    search_radius: u32,
}

/// Push constants for the flow_convert shader (f16 → f32 conversion).
#[repr(C)]
#[derive(Clone, Copy)]
struct FlowConvertPc {
    count: u32,
}

/// NVIDIA optical flow frame interpolator using VK_NV_optical_flow extension.
///
/// Uses NVIDIA's hardware-accelerated optical flow engine (available on
/// Turing and newer GPUs) for high-quality motion estimation, combined
/// with a Vulkan compute warp/blend pass for the actual interpolation.
///
/// The VK_NV_optical_flow extension provides dedicated hardware units
/// for motion estimation that are significantly faster and higher quality
/// than software block matching.
pub struct NvidiaOpticalFlowInterpolator {
    ctx: Arc<Mutex<VulkanContext>>,
    // Optical flow extension function pointers.
    of_fns: NvOpticalFlowFns,
    // flow_convert pipeline: R16G16_SFLOAT (f16 pairs) → vec2<f32>.
    convert_pipeline: vk::Pipeline,
    convert_pipeline_layout: vk::PipelineLayout,
    convert_desc_layout: vk::DescriptorSetLayout,
    // Warp/blend compute pipeline (same as FSR2).
    warp_pipeline: vk::Pipeline,
    warp_pipeline_layout: vk::PipelineLayout,
    warp_desc_layout: vk::DescriptorSetLayout,
    desc_pool: vk::DescriptorPool,
    cached: Mutex<Option<NvOfResources>>,
}

/// VK_NV_optical_flow extension function pointers.
struct NvOpticalFlowFns {
    create_session: vk::PFN_vkCreateOpticalFlowSessionNV,
    destroy_session: vk::PFN_vkDestroyOpticalFlowSessionNV,
    bind_image: vk::PFN_vkBindOpticalFlowSessionImageNV,
    cmd_execute: vk::PFN_vkCmdOpticalFlowExecuteNV,
}

struct NvOfResources {
    width: u32,
    height: u32,
    // Per-resolution OF session (recreated on dimension change).
    of_session: vk::OpticalFlowSessionNV,
    // Input images for optical flow.
    image_a: vk::Image,
    image_a_mem: vk::DeviceMemory,
    image_a_view: vk::ImageView,
    image_b: vk::Image,
    image_b_mem: vk::DeviceMemory,
    image_b_view: vk::ImageView,
    // Output flow vector image (R16G16_SFLOAT, quarter resolution).
    flow_image: vk::Image,
    flow_image_mem: vk::DeviceMemory,
    flow_image_view: vk::ImageView,
    // Raw flow data buffer (R16G16_SFLOAT copied from flow_image, f16 pairs).
    flow_raw_buf: vk::Buffer,
    flow_raw_mem: vk::DeviceMemory,
    // Frame buffers for warp/blend compute pass.
    frame_a_buf: vk::Buffer,
    frame_a_mem: vk::DeviceMemory,
    frame_b_buf: vk::Buffer,
    frame_b_mem: vk::DeviceMemory,
    // Motion buffer: vec2<f32> per block, written by flow_convert shader.
    motion_buf: vk::Buffer,
    motion_mem: vk::DeviceMemory,
    output_buf: vk::Buffer,
    output_mem: vk::DeviceMemory,
    // Staging buffers for CPU → GPU upload (two for parallel upload).
    staging_a_buf: vk::Buffer,
    staging_a_mem: vk::DeviceMemory,
    staging_b_buf: vk::Buffer,
    staging_b_mem: vk::DeviceMemory,
    readback_buf: vk::Buffer,
    readback_mem: vk::DeviceMemory,
    convert_desc_set: vk::DescriptorSet,
    warp_desc_set: vk::DescriptorSet,
    cmd: vk::CommandBuffer,
}

impl NvidiaOpticalFlowInterpolator {
    /// Create a new NVIDIA optical flow interpolator.
    ///
    /// Requires a Turing or newer NVIDIA GPU with VK_NV_optical_flow support.
    pub fn new() -> Result<Self, InterpolateError> {
        let ext_name = c"VK_NV_optical_flow";
        let ctx = Arc::new(Mutex::new(VulkanContext::new(&[ext_name])?));
        let guard = ctx.lock().unwrap();

        // Verify the device is NVIDIA and has the extension.
        if guard.vendor_id != 0x10DE {
            return Err(InterpolateError::InitFailed(
                "NVIDIA optical flow requires an NVIDIA GPU".into(),
            ));
        }

        let available_exts = unsafe {
            guard
                .instance
                .enumerate_device_extension_properties(guard.physical_device)
        }
        .unwrap_or_default();
        let has_of = available_exts.iter().any(|e| {
            let name = unsafe { CStr::from_ptr(e.extension_name.as_ptr()) };
            name == ext_name
        });
        if !has_of {
            return Err(InterpolateError::InitFailed(
                "VK_NV_optical_flow extension not available (requires Turing+ GPU)".into(),
            ));
        }

        // Load extension function pointers via get_device_proc_addr.
        // Safety: the extension was confirmed available above; transmute is valid
        // because the Vulkan spec guarantees function signatures for known extension names.
        let of_fns = {
            let load_fn = |name: &CStr| -> Result<unsafe extern "system" fn(), InterpolateError> {
                unsafe {
                    guard
                        .instance
                        .get_device_proc_addr(guard.device.handle(), name.as_ptr())
                }
                .ok_or_else(|| {
                    InterpolateError::InitFailed(format!("{} not found", name.to_string_lossy()))
                })
            };

            let create_fn = load_fn(c"vkCreateOpticalFlowSessionNV")?;
            let destroy_fn = load_fn(c"vkDestroyOpticalFlowSessionNV")?;
            let bind_fn = load_fn(c"vkBindOpticalFlowSessionImageNV")?;
            let exec_fn = load_fn(c"vkCmdOpticalFlowExecuteNV")?;

            // Safety: function pointer signatures are guaranteed by the Vulkan spec
            // for these well-known extension entry points.
            #[allow(clippy::missing_transmute_annotations)]
            unsafe {
                NvOpticalFlowFns {
                    create_session: std::mem::transmute(create_fn),
                    destroy_session: std::mem::transmute(destroy_fn),
                    bind_image: std::mem::transmute(bind_fn),
                    cmd_execute: std::mem::transmute(exec_fn),
                }
            }
        };

        // --- flow_convert pipeline: f16 → f32 motion vector conversion ---
        let convert_spv = ash::util::read_spv(&mut std::io::Cursor::new(FLOW_CONVERT_SPV))
            .map_err(|e| InterpolateError::InitFailed(format!("convert SPIR-V: {e}")))?;
        let convert_module_info = vk::ShaderModuleCreateInfo::default().code(&convert_spv);
        let convert_module = unsafe {
            guard
                .device
                .create_shader_module(&convert_module_info, None)
        }
        .map_err(|e| InterpolateError::InitFailed(format!("convert shader: {e}")))?;

        // Convert descriptor layout: flow_raw(0), motion(1)
        let convert_bindings = [
            desc_binding(0, vk::DescriptorType::STORAGE_BUFFER),
            desc_binding(1, vk::DescriptorType::STORAGE_BUFFER),
        ];
        let convert_desc_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&convert_bindings);
        let convert_desc_layout = unsafe {
            guard
                .device
                .create_descriptor_set_layout(&convert_desc_layout_info, None)
        }
        .map_err(|e| InterpolateError::InitFailed(format!("convert desc layout: {e}")))?;

        let convert_push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<FlowConvertPc>() as u32);
        let convert_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&convert_desc_layout))
            .push_constant_ranges(std::slice::from_ref(&convert_push_range));
        let convert_pipeline_layout = unsafe {
            guard
                .device
                .create_pipeline_layout(&convert_layout_info, None)
        }
        .map_err(|e| InterpolateError::InitFailed(format!("convert layout: {e}")))?;

        let convert_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(convert_module)
            .name(c"main");
        let convert_pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(convert_stage)
            .layout(convert_pipeline_layout);

        // --- warp/blend pipeline ---
        let warp_spv = ash::util::read_spv(&mut std::io::Cursor::new(WARP_BLEND_SPV))
            .map_err(|e| InterpolateError::InitFailed(format!("warp SPIR-V: {e}")))?;
        let warp_module_info = vk::ShaderModuleCreateInfo::default().code(&warp_spv);
        let warp_module = unsafe { guard.device.create_shader_module(&warp_module_info, None) }
            .map_err(|e| InterpolateError::InitFailed(format!("warp shader: {e}")))?;

        // Warp descriptor layout: frame_a(0), frame_b(1), motion(2), output(3)
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

        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<PushConstants>() as u32);
        let warp_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&warp_desc_layout))
            .push_constant_ranges(std::slice::from_ref(&push_range));
        let warp_pipeline_layout =
            unsafe { guard.device.create_pipeline_layout(&warp_layout_info, None) }
                .map_err(|e| InterpolateError::InitFailed(format!("pipeline layout: {e}")))?;

        let warp_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(warp_module)
            .name(c"main");
        let warp_pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(warp_stage)
            .layout(warp_pipeline_layout);

        // Create both pipelines in one call.
        let pipelines = unsafe {
            guard.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[convert_pipeline_info, warp_pipeline_info],
                None,
            )
        }
        .map_err(|(_p, e)| InterpolateError::InitFailed(format!("pipelines: {e}")))?;
        let convert_pipeline = pipelines[0];
        let warp_pipeline = pipelines[1];

        unsafe {
            guard.device.destroy_shader_module(convert_module, None);
            guard.device.destroy_shader_module(warp_module, None);
        }

        // Descriptor pool: 2 for convert + 4 for warp = 6 storage buffer descriptors, 2 sets.
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 12,
        }];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(4)
            .pool_sizes(&pool_sizes);
        let desc_pool = unsafe { guard.device.create_descriptor_pool(&pool_info, None) }
            .map_err(|e| InterpolateError::InitFailed(format!("desc pool: {e}")))?;

        drop(guard);

        Ok(Self {
            ctx,
            of_fns,
            convert_pipeline,
            convert_pipeline_layout,
            convert_desc_layout,
            warp_pipeline,
            warp_pipeline_layout,
            warp_desc_layout,
            desc_pool,
            cached: Mutex::new(None),
        })
    }

    fn ensure_resources(&self, width: u32, height: u32) -> Result<(), InterpolateError> {
        let mut cache = self.cached.lock().unwrap();
        if let Some(ref r) = *cache {
            if r.width == width && r.height == height {
                return Ok(());
            }
            let guard = self.ctx.lock().unwrap();
            destroy_resources(&guard.device, &self.of_fns, cache.take().unwrap());
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
        let host_visible =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let device_local = vk::MemoryPropertyFlags::DEVICE_LOCAL;

        // Create OF session at actual frame resolution.
        let session_info = vk::OpticalFlowSessionCreateInfoNV::default()
            .width(width)
            .height(height)
            .image_format(vk::Format::R8G8B8A8_UNORM)
            .flow_vector_format(vk::Format::R16G16_SFLOAT)
            .output_grid_size(vk::OpticalFlowGridSizeFlagsNV::TYPE_4X4);

        let mut of_session = vk::OpticalFlowSessionNV::null();
        let result = unsafe {
            (self.of_fns.create_session)(
                guard.device.handle(),
                &session_info,
                std::ptr::null(),
                &mut of_session,
            )
        };
        if result != vk::Result::SUCCESS {
            return Err(InterpolateError::InterpolateFailed(format!(
                "vkCreateOpticalFlowSessionNV failed: {result:?}"
            )));
        }

        // Create input images for optical flow (R8G8B8A8_UNORM).
        let (image_a, image_a_mem, image_a_view) =
            create_of_image(&guard, width, height, vk::Format::R8G8B8A8_UNORM)?;
        let (image_b, image_b_mem, image_b_view) =
            create_of_image(&guard, width, height, vk::Format::R8G8B8A8_UNORM)?;

        // Flow vector output image (R16G16_SFLOAT, at 4x4 grid = quarter resolution).
        let flow_w = width.div_ceil(4);
        let flow_h = height.div_ceil(4);
        let (flow_image, flow_image_mem, flow_image_view) =
            create_of_image(&guard, flow_w, flow_h, vk::Format::R16G16_SFLOAT)?;

        // Bind images to optical flow session.
        unsafe {
            let _ = (self.of_fns.bind_image)(
                guard.device.handle(),
                of_session,
                vk::OpticalFlowSessionBindingPointNV::INPUT,
                image_a_view,
                vk::ImageLayout::GENERAL,
            );
            let _ = (self.of_fns.bind_image)(
                guard.device.handle(),
                of_session,
                vk::OpticalFlowSessionBindingPointNV::REFERENCE,
                image_b_view,
                vk::ImageLayout::GENERAL,
            );
            let _ = (self.of_fns.bind_image)(
                guard.device.handle(),
                of_session,
                vk::OpticalFlowSessionBindingPointNV::FLOW_VECTOR,
                flow_image_view,
                vk::ImageLayout::GENERAL,
            );
        }

        // Frame buffers for warp/blend pass.
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        let try_create = |size: u64,
                          usage: vk::BufferUsageFlags|
         -> Result<(vk::Buffer, vk::DeviceMemory), InterpolateError> {
            guard
                .create_buffer(size, usage, device_local)
                .or_else(|_| guard.create_buffer(size, usage, host_visible))
        };

        let block_size = 4u32; // 4x4 grid matches OF output
        let blocks_x = width.div_ceil(block_size);
        let blocks_y = height.div_ceil(block_size);
        let flow_count = (blocks_x * blocks_y) as u64;
        let flow_raw_size = flow_count * 4; // R16G16_SFLOAT = 4 bytes per pixel
        let motion_size = flow_count * 8; // vec2<f32> = 8 bytes per block

        // flow_raw_buf: receives raw f16 data from vkCmdCopyImageToBuffer.
        let (flow_raw_buf, flow_raw_mem) = try_create(
            flow_raw_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        )?;
        let (frame_a_buf, frame_a_mem2) = try_create(frame_size, storage)?;
        let (frame_b_buf, frame_b_mem2) = try_create(frame_size, storage)?;
        // motion_buf: written by flow_convert shader (f32 pairs).
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

        // Allocate descriptor sets: convert + warp.
        let layouts = [self.convert_desc_layout, self.warp_desc_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.desc_pool)
            .set_layouts(&layouts);
        let desc_sets = unsafe { guard.device.allocate_descriptor_sets(&alloc_info) }
            .map_err(|e| InterpolateError::InterpolateFailed(format!("alloc desc: {e}")))?;

        // Write convert descriptor set: flow_raw(0) → motion(1).
        let convert_bufs = [
            vk::DescriptorBufferInfo::default()
                .buffer(flow_raw_buf)
                .range(flow_raw_size),
            vk::DescriptorBufferInfo::default()
                .buffer(motion_buf)
                .range(motion_size),
        ];
        // Write warp descriptor set: frame_a(0), frame_b(1), motion(2), output(3).
        let warp_bufs = [
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
            // convert set
            vk::WriteDescriptorSet::default()
                .dst_set(desc_sets[0])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&convert_bufs[0..1]),
            vk::WriteDescriptorSet::default()
                .dst_set(desc_sets[0])
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&convert_bufs[1..2]),
            // warp set
            vk::WriteDescriptorSet::default()
                .dst_set(desc_sets[1])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&warp_bufs[0..1]),
            vk::WriteDescriptorSet::default()
                .dst_set(desc_sets[1])
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&warp_bufs[1..2]),
            vk::WriteDescriptorSet::default()
                .dst_set(desc_sets[1])
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&warp_bufs[2..3]),
            vk::WriteDescriptorSet::default()
                .dst_set(desc_sets[1])
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&warp_bufs[3..4]),
        ];
        unsafe { guard.device.update_descriptor_sets(&writes, &[]) };

        let cmd = guard.allocate_command_buffer()?;

        *cache = Some(NvOfResources {
            width,
            height,
            of_session,
            image_a,
            image_a_mem,
            image_a_view,
            image_b,
            image_b_mem,
            image_b_view,
            flow_image,
            flow_image_mem,
            flow_image_view,
            flow_raw_buf,
            flow_raw_mem,
            frame_a_buf,
            frame_a_mem: frame_a_mem2,
            frame_b_buf,
            frame_b_mem: frame_b_mem2,
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
            convert_desc_set: desc_sets[0],
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

        // Upload both frames to separate staging buffers (CPU map, no GPU wait).
        guard.upload_to_buffer(res.staging_a_mem, &a.data)?;
        guard.upload_to_buffer(res.staging_b_mem, &b.data)?;

        let cmd = res.cmd;

        let flow_w = w.div_ceil(4);
        let flow_h = h.div_ceil(4);
        let block_size = 4u32;

        let subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let subresource_layers = vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        };

        // Single command buffer: upload buffers + upload images + OF + flow→motion + warp + readback.
        unsafe {
            guard
                .device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .map_err(|e| InterpolateError::InterpolateFailed(format!("reset cmd: {e}")))?;
            guard
                .device
                .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
                .map_err(|e| InterpolateError::InterpolateFailed(format!("begin cmd: {e}")))?;

            // 1) Copy staging → frame buffers (for warp/blend compute pass).
            let buf_copy = vk::BufferCopy::default().size(frame_size);
            guard
                .device
                .cmd_copy_buffer(cmd, res.staging_a_buf, res.frame_a_buf, &[buf_copy]);
            guard
                .device
                .cmd_copy_buffer(cmd, res.staging_b_buf, res.frame_b_buf, &[buf_copy]);

            // 2) Transition OF input images: UNDEFINED → TRANSFER_DST_OPTIMAL.
            let to_dst = |image: vk::Image| {
                vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .image(image)
                    .subresource_range(subresource_range)
            };

            let img_barriers = [to_dst(res.image_a), to_dst(res.image_b)];
            guard.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &img_barriers,
            );

            // 3) Copy staging buffers → OF input images.
            let img_copy = vk::BufferImageCopy::default()
                .image_subresource(subresource_layers)
                .image_extent(vk::Extent3D {
                    width: w,
                    height: h,
                    depth: 1,
                });
            guard.device.cmd_copy_buffer_to_image(
                cmd,
                res.staging_a_buf,
                res.image_a,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[img_copy],
            );
            guard.device.cmd_copy_buffer_to_image(
                cmd,
                res.staging_b_buf,
                res.image_b,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[img_copy],
            );

            // 4) Transition OF input images: TRANSFER_DST → GENERAL (for OF read).
            //    Transition flow image: UNDEFINED → GENERAL (for OF write).
            let to_general = |image: vk::Image,
                              old: vk::ImageLayout,
                              src_access: vk::AccessFlags| {
                vk::ImageMemoryBarrier::default()
                    .src_access_mask(src_access)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                    .old_layout(old)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(image)
                    .subresource_range(subresource_range)
            };
            let pre_of_barriers = [
                to_general(
                    res.image_a,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::AccessFlags::TRANSFER_WRITE,
                ),
                to_general(
                    res.image_b,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::AccessFlags::TRANSFER_WRITE,
                ),
                to_general(
                    res.flow_image,
                    vk::ImageLayout::UNDEFINED,
                    vk::AccessFlags::empty(),
                ),
            ];
            guard.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &pre_of_barriers,
            );

            // 5) Execute hardware optical flow.
            let exec_info = vk::OpticalFlowExecuteInfoNV::default();
            (self.of_fns.cmd_execute)(cmd, res.of_session, &exec_info);

            // 6) Barrier: OF write → transfer read (for flow_image → motion_buf copy).
            let post_of_barrier = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .image(res.flow_image)
                .subresource_range(subresource_range);
            guard.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[post_of_barrier],
            );

            // 7) Copy flow_image (R16G16_SFLOAT, quarter res) → flow_raw_buf.
            let flow_copy = vk::BufferImageCopy::default()
                .image_subresource(subresource_layers)
                .image_extent(vk::Extent3D {
                    width: flow_w,
                    height: flow_h,
                    depth: 1,
                });
            guard.device.cmd_copy_image_to_buffer(
                cmd,
                res.flow_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                res.flow_raw_buf,
                &[flow_copy],
            );

            // 8) Barrier: transfer → compute (flow_convert reads flow_raw_buf).
            let pre_convert = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            guard.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[pre_convert],
                &[],
                &[],
            );

            // 9) flow_convert: f16 flow pairs → f32 motion vectors.
            let flow_count = flow_w * flow_h;
            let convert_pc = FlowConvertPc { count: flow_count };
            // Safety: FlowConvertPc is #[repr(C)] with a single u32 field.
            let convert_pc_bytes = std::slice::from_raw_parts(
                &convert_pc as *const FlowConvertPc as *const u8,
                std::mem::size_of::<FlowConvertPc>(),
            );
            guard.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.convert_pipeline,
            );
            guard.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.convert_pipeline_layout,
                0,
                &[res.convert_desc_set],
                &[],
            );
            guard.device.cmd_push_constants(
                cmd,
                self.convert_pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                convert_pc_bytes,
            );
            guard
                .device
                .cmd_dispatch(cmd, flow_count.div_ceil(256), 1, 1);

            // 10) Barrier: flow_convert write → warp/blend read.
            let pre_warp = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            guard.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[pre_warp],
                &[],
                &[],
            );

            // 11) Warp + blend using f32 motion vectors.
            let pc = PushConstants {
                width: w,
                height: h,
                t,
                block_size,
                search_radius: 0,
            };
            // Safety: PushConstants is #[repr(C)] with no padding holes; reading
            // its memory as &[u8] is safe for push constant upload.
            let pc_bytes = std::slice::from_raw_parts(
                &pc as *const PushConstants as *const u8,
                std::mem::size_of::<PushConstants>(),
            );

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

            // 12) Barrier: compute → transfer (readback).
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

            // 13) Copy output → readback buffer.
            guard
                .device
                .cmd_copy_buffer(cmd, res.output_buf, res.readback_buf, &[buf_copy]);

            guard
                .device
                .end_command_buffer(cmd)
                .map_err(|e| InterpolateError::InterpolateFailed(format!("end cmd: {e}")))?;
        }

        // Single submit + single fence wait.
        guard.submit_and_wait(cmd)?;
        guard.read_from_buffer(res.readback_mem, frame_size as usize)
    }
}

impl FrameInterpolator for NvidiaOpticalFlowInterpolator {
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
        // Hardware OF is very fast (~1ms), warp/blend adds ~1ms.
        2.0
    }

    fn name(&self) -> &str {
        "nvidia-optical-flow"
    }
}

impl Drop for NvidiaOpticalFlowInterpolator {
    fn drop(&mut self) {
        let guard = self.ctx.lock().unwrap();
        if let Some(res) = self.cached.lock().unwrap().take() {
            destroy_resources(&guard.device, &self.of_fns, res);
        }
        unsafe {
            guard.device.destroy_descriptor_pool(self.desc_pool, None);
            guard.device.destroy_pipeline(self.convert_pipeline, None);
            guard.device.destroy_pipeline(self.warp_pipeline, None);
            guard
                .device
                .destroy_pipeline_layout(self.convert_pipeline_layout, None);
            guard
                .device
                .destroy_pipeline_layout(self.warp_pipeline_layout, None);
            guard
                .device
                .destroy_descriptor_set_layout(self.convert_desc_layout, None);
            guard
                .device
                .destroy_descriptor_set_layout(self.warp_desc_layout, None);
        }
    }
}

fn create_of_image(
    ctx: &VulkanContext,
    width: u32,
    height: u32,
    format: vk::Format,
) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView), InterpolateError> {
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(
            vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::STORAGE,
        )
        .initial_layout(vk::ImageLayout::UNDEFINED);

    let image = unsafe { ctx.device.create_image(&image_info, None) }
        .map_err(|e| InterpolateError::InterpolateFailed(format!("create image: {e}")))?;

    let mem_req = unsafe { ctx.device.get_image_memory_requirements(image) };
    let mem_props = unsafe {
        ctx.instance
            .get_physical_device_memory_properties(ctx.physical_device)
    };

    let mem_type_index = (0..mem_props.memory_type_count)
        .find(|&i| {
            let type_bits = mem_req.memory_type_bits & (1 << i);
            let prop_match = mem_props.memory_types[i as usize]
                .property_flags
                .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL);
            type_bits != 0 && prop_match
        })
        .ok_or_else(|| {
            unsafe { ctx.device.destroy_image(image, None) };
            InterpolateError::InterpolateFailed("no suitable image memory type".into())
        })?;

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_req.size)
        .memory_type_index(mem_type_index);
    let memory = unsafe { ctx.device.allocate_memory(&alloc_info, None) }.map_err(|e| {
        unsafe { ctx.device.destroy_image(image, None) };
        InterpolateError::InterpolateFailed(format!("allocate image memory: {e}"))
    })?;

    unsafe { ctx.device.bind_image_memory(image, memory, 0) }.map_err(|e| {
        unsafe {
            ctx.device.free_memory(memory, None);
            ctx.device.destroy_image(image, None);
        }
        InterpolateError::InterpolateFailed(format!("bind image memory: {e}"))
    })?;

    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    let view = unsafe { ctx.device.create_image_view(&view_info, None) }.map_err(|e| {
        unsafe {
            ctx.device.free_memory(memory, None);
            ctx.device.destroy_image(image, None);
        }
        InterpolateError::InterpolateFailed(format!("create image view: {e}"))
    })?;

    Ok((image, memory, view))
}

fn destroy_resources(device: &ash::Device, of_fns: &NvOpticalFlowFns, res: NvOfResources) {
    unsafe {
        (of_fns.destroy_session)(device.handle(), res.of_session, std::ptr::null());
        device.destroy_image_view(res.image_a_view, None);
        device.destroy_image(res.image_a, None);
        device.free_memory(res.image_a_mem, None);
        device.destroy_image_view(res.image_b_view, None);
        device.destroy_image(res.image_b, None);
        device.free_memory(res.image_b_mem, None);
        device.destroy_image_view(res.flow_image_view, None);
        device.destroy_image(res.flow_image, None);
        device.free_memory(res.flow_image_mem, None);
        device.destroy_buffer(res.flow_raw_buf, None);
        device.free_memory(res.flow_raw_mem, None);
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

// Safety: All Vulkan handles are opaque integers. The NvOpticalFlowFns struct
// contains raw function pointers loaded from the Vulkan driver. All Vulkan calls
// are serialized through the Arc<Mutex<VulkanContext>>.
unsafe impl Send for NvidiaOpticalFlowInterpolator {}
unsafe impl Sync for NvidiaOpticalFlowInterpolator {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvidia_of_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<NvidiaOpticalFlowInterpolator>();
    }

    #[test]
    fn push_constants_size() {
        assert_eq!(std::mem::size_of::<PushConstants>(), 20);
        assert_eq!(std::mem::size_of::<FlowConvertPc>(), 4);
    }

    #[test]
    #[ignore] // requires NVIDIA Turing+ GPU
    fn nvidia_of_init() {
        let interp = NvidiaOpticalFlowInterpolator::new();
        if let Ok(interp) = interp {
            assert_eq!(interp.name(), "nvidia-optical-flow");
        }
    }

    #[test]
    #[ignore] // requires NVIDIA Turing+ GPU
    fn nvidia_of_interpolate_small() {
        let interp = match NvidiaOpticalFlowInterpolator::new() {
            Ok(i) => i,
            Err(_) => return,
        };
        let a = GpuFrame::from_data(vec![0u8; 64 * 64 * 4], 64, 64, 256, 0);
        let b = GpuFrame::from_data(vec![128u8; 64 * 64 * 4], 64, 64, 256, 1000);
        let result = interp.interpolate(&a, &b, 0.5).unwrap();
        assert_eq!(result.width, 64);
        assert_eq!(result.data.len(), 64 * 64 * 4);
    }
}

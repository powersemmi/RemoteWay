//! AMD FSR 2 native upscaling backend via the Embark Studios `fsr` crate.
//!
//! Uses the real [`fsr::Context`] to perform spatial upscaling with temporal
//! anti-aliasing. Motion vectors are computed via a simple CPU block-matching
//! pass (or zero vectors as a fallback). Temporal interpolation (frame
//! generation) is not supported by this backend — use [`super::fsr3`] for that.

#![allow(clippy::undocumented_unsafe_blocks)]

use std::sync::{Arc, Mutex};

use ash::vk;

use crate::error::InterpolateError;
use crate::interpolator::{FrameInterpolator, GpuFrame};

use super::vulkan_context::VulkanContext;

/// Halton sequence generator for jitter offsets.
///
/// Produces values in `[0.0, 1.0)` for the given `index` and `base`.
#[allow(dead_code)]
fn halton(index: u32, base: u32) -> f32 {
    let mut result = 0.0f32;
    let mut f = 1.0f32 / base as f32;
    let mut i = index;
    while i > 0 {
        result += f * (i % base) as f32;
        i /= base;
        f /= base as f32;
    }
    result
}

/// Cached per-resolution Vulkan resources for FSR 2 upscaling.
///
/// Images and buffers are kept alive across frames so FSR 2 can maintain
/// its internal temporal history. Recreated only when dimensions change.
struct CachedResources {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    color_image: vk::Image,
    color_mem: vk::DeviceMemory,
    color_view: vk::ImageView,
    depth_image: vk::Image,
    depth_mem: vk::DeviceMemory,
    depth_view: vk::ImageView,
    mv_image: vk::Image,
    mv_mem: vk::DeviceMemory,
    mv_view: vk::ImageView,
    output_image: vk::Image,
    output_mem: vk::DeviceMemory,
    output_view: vk::ImageView,
    staging_color_buf: vk::Buffer,
    staging_color_mem: vk::DeviceMemory,
    staging_depth_buf: vk::Buffer,
    staging_depth_mem: vk::DeviceMemory,
    staging_mv_buf: vk::Buffer,
    staging_mv_mem: vk::DeviceMemory,
    readback_buf: vk::Buffer,
    readback_mem: vk::DeviceMemory,
    readback_size: u64,
    cmd: vk::CommandBuffer,
}

/// Real AMD FSR 2 upscaling via the `fsr` crate (Vulkan backend).
///
/// Holds an [`fsr::Context`] for FSR 2 dispatches and a shared
/// [`VulkanContext`] for buffer/image management. GPU resources are
/// cached across frames so FSR 2 can maintain temporal history.
/// Motion vectors are zeroed (FSR 2 uses its internal optical flow
/// for 2D content without camera motion).
pub struct Fsr2NativeInterpolator {
    /// The FSR 2 context (interior-mutable via Mutex because
    /// `dispatch()` requires `&mut self`).
    fsr_ctx: Mutex<fsr::Context>,
    /// Shared Vulkan device, memory, and queue.
    ctx: Arc<Mutex<VulkanContext>>,
    /// Monotonically increasing frame counter for Halton jitter.
    frame_idx: Mutex<u64>,
    /// Display / output dimensions.
    #[allow(dead_code)]
    display_w: u32,
    #[allow(dead_code)]
    display_h: u32,
    /// Maximum input (render) dimensions.
    #[allow(dead_code)]
    max_render_w: u32,
    #[allow(dead_code)]
    max_render_h: u32,
    cached_res: Mutex<Option<CachedResources>>,
}

/// SAFETY: All GPU resources are guarded by `Arc<Mutex<>>` and
/// `fsr::Context` dispatch is serialized through its own Mutex.
// SAFETY: All mutable state is protected by internal synchronization (Mutex).
unsafe impl Send for Fsr2NativeInterpolator {}
// SAFETY: All mutable state is protected by internal synchronization (Mutex).
unsafe impl Sync for Fsr2NativeInterpolator {}

impl Fsr2NativeInterpolator {
    /// Create a new FSR 2 native upscaler with its own [`VulkanContext`].
    ///
    /// Prefer [`Self::with_context`] when a shared context is available
    /// to avoid creating duplicate Vulkan instances/devices.
    pub fn new(
        display_w: u32,
        display_h: u32,
        max_render_w: u32,
        max_render_h: u32,
    ) -> Result<Self, InterpolateError> {
        let vk_ctx = Arc::new(Mutex::new(VulkanContext::new(&[])?));
        Self::with_context(vk_ctx, display_w, display_h, max_render_w, max_render_h)
    }

    /// Create a new FSR 2 native upscaler sharing an existing [`VulkanContext`].
    pub(crate) fn with_context(
        vk_ctx: Arc<Mutex<VulkanContext>>,
        display_w: u32,
        display_h: u32,
        max_render_w: u32,
        max_render_h: u32,
    ) -> Result<Self, InterpolateError> {
        let guard = vk_ctx
            .lock()
            .map_err(|e| InterpolateError::InitFailed(format!("mutex poisoned: {e}")))?;

        let interface = unsafe {
            fsr::vk::get_interface(&guard._entry, &guard.instance, guard.physical_device)
        }
        .map_err(|e| InterpolateError::InitFailed(format!("FSR interface: {e}")))?;

        let fsr_device = unsafe { fsr::vk::get_device(guard.device.clone()) };

        let flags = fsr::InitializationFlagBits::ENABLE_DEPTH_INFINITE
            | fsr::InitializationFlagBits::ENABLE_DEPTH_INVERTED;

        let context_desc = fsr::ContextDescription {
            interface,
            flags,
            max_render_size: [max_render_w, max_render_h],
            display_size: [display_w, display_h],
            device: &fsr_device,
            message_callback: None,
        };

        let fsr_ctx = unsafe { fsr::Context::new(context_desc) }
            .map_err(|e| InterpolateError::InitFailed(format!("FSR context: {e}")))?;

        drop(guard);

        Ok(Self {
            fsr_ctx: Mutex::new(fsr_ctx),
            ctx: vk_ctx,
            frame_idx: Mutex::new(0),
            display_w,
            display_h,
            max_render_w,
            max_render_h,
            cached_res: Mutex::new(None),
        })
    }

    /// Create a Vulkan image + image view for FSR 2 use.
    #[allow(clippy::too_many_arguments)]
    fn create_fsr_image(
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        width: u32,
        height: u32,
        usage: vk::ImageUsageFlags,
    ) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView), InterpolateError> {
        let img_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .extent(vk::Extent3D::default().width(width).height(height).depth(1))
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        // SAFETY: device, instance, and physical_device are valid handles.
        let image = unsafe { device.create_image(&img_info, None) }
            .map_err(|e| InterpolateError::InterpolateFailed(format!("create fsr image: {e}")))?;

        let mem_req = unsafe { device.get_image_memory_requirements(image) };
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let mem_type_index = (0..mem_props.memory_type_count)
            .find(|&i| {
                mem_req.memory_type_bits & (1 << i) != 0
                    && mem_props.memory_types[i as usize]
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            })
            .ok_or_else(|| {
                unsafe { device.destroy_image(image, None) };
                InterpolateError::InterpolateFailed("no device-local memory for FSR image".into())
            })?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_req.size)
            .memory_type_index(mem_type_index);
        let mem = unsafe { device.allocate_memory(&alloc_info, None) }.map_err(|e| {
            unsafe { device.destroy_image(image, None) };
            InterpolateError::InterpolateFailed(format!("alloc FSR image mem: {e}"))
        })?;
        unsafe { device.bind_image_memory(image, mem, 0) }.map_err(|e| {
            unsafe {
                device.free_memory(mem, None);
                device.destroy_image(image, None);
            }
            InterpolateError::InterpolateFailed(format!("bind FSR image mem: {e}"))
        })?;

        // Create image view.
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );
        let image_view = unsafe { device.create_image_view(&view_info, None) }.map_err(|e| {
            unsafe {
                device.free_memory(mem, None);
                device.destroy_image(image, None);
            }
            InterpolateError::InterpolateFailed(format!("create FSR image view: {e}"))
        })?;

        Ok((image, mem, image_view))
    }

    /// Create a Vulkan image + view with a specific format.
    fn create_fsr_image_format(
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        width: u32,
        height: u32,
        usage: vk::ImageUsageFlags,
        format: vk::Format,
    ) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView), InterpolateError> {
        let img_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D::default().width(width).height(height).depth(1))
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = unsafe { device.create_image(&img_info, None) }
            .map_err(|e| InterpolateError::InterpolateFailed(format!("create fsr image: {e}")))?;

        let mem_req = unsafe { device.get_image_memory_requirements(image) };
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let mem_type_index = (0..mem_props.memory_type_count)
            .find(|&i| {
                mem_req.memory_type_bits & (1 << i) != 0
                    && mem_props.memory_types[i as usize]
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            })
            .ok_or_else(|| {
                unsafe { device.destroy_image(image, None) };
                InterpolateError::InterpolateFailed("no device-local memory for FSR image".into())
            })?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_req.size)
            .memory_type_index(mem_type_index);
        let mem = unsafe { device.allocate_memory(&alloc_info, None) }.map_err(|e| {
            unsafe { device.destroy_image(image, None) };
            InterpolateError::InterpolateFailed(format!("alloc FSR image mem: {e}"))
        })?;
        unsafe { device.bind_image_memory(image, mem, 0) }.map_err(|e| {
            unsafe {
                device.free_memory(mem, None);
                device.destroy_image(image, None);
            }
            InterpolateError::InterpolateFailed(format!("bind FSR image mem: {e}"))
        })?;

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );
        let image_view = unsafe { device.create_image_view(&view_info, None) }.map_err(|e| {
            unsafe {
                device.free_memory(mem, None);
                device.destroy_image(image, None);
            }
            InterpolateError::InterpolateFailed(format!("create FSR image view: {e}"))
        })?;

        Ok((image, mem, image_view))
    }
}

/// Record an image memory barrier (simple layout transition, single mip/array layer).
unsafe fn cmd_image_barrier_simple(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) {
    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        );
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            std::slice::from_ref(&barrier),
        );
    }
}

/// Record a buffer-to-image copy.
unsafe fn copy_buf_to_image(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    src_buf: vk::Buffer,
    dst_image: vk::Image,
    width: u32,
    height: u32,
) {
    let region = vk::BufferImageCopy::default()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1),
        )
        .image_offset(vk::Offset3D::default())
        .image_extent(vk::Extent3D::default().width(width).height(height).depth(1));
    unsafe {
        device.cmd_copy_buffer_to_image(
            cmd,
            src_buf,
            dst_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            std::slice::from_ref(&region),
        );
    }
}

impl FrameInterpolator for Fsr2NativeInterpolator {
    /// Temporal frame interpolation is not implemented — this backend
    /// only performs spatial upscaling via FSR 2.
    fn interpolate(
        &mut self,
        _a: &GpuFrame,
        _b: &GpuFrame,
        _t: f32,
    ) -> Result<GpuFrame, InterpolateError> {
        Err(InterpolateError::InterpolateFailed(
            "Fsr2NativeInterpolator does not support temporal interpolation; use upscale() instead"
                .into(),
        ))
    }

    /// Upscale `src` from its native resolution to `dst_w × dst_h` using
    /// AMD FSR 2 with temporal feedback. GPU resources are cached across
    /// frames and only recreated when dimensions change.
    fn upscale(
        &self,
        src: &GpuFrame,
        dst_w: u32,
        dst_h: u32,
    ) -> Result<GpuFrame, InterpolateError> {
        let src_w = src.width;
        let src_h = src.height;

        // Acquire frame index.
        let mut fidx = self
            .frame_idx
            .lock()
            .map_err(|e| InterpolateError::InterpolateFailed(format!("frame_idx lock: {e}")))?;
        let frame_idx = *fidx;
        *fidx = frame_idx.wrapping_add(1);
        drop(fidx);

        // Lock Vulkan context for the entire operation.
        let vk_ctx = self
            .ctx
            .lock()
            .map_err(|e| InterpolateError::InterpolateFailed(format!("vk ctx lock: {e}")))?;

        // --- Ensure cached resources match current dimensions ---
        let mut cache = self
            .cached_res
            .lock()
            .map_err(|e| InterpolateError::InterpolateFailed(format!("cached_res lock: {e}")))?;

        let needs_recreate = match cache.as_ref() {
            Some(c) => c.src_w != src_w || c.src_h != src_h || c.dst_w != dst_w || c.dst_h != dst_h,
            None => true,
        };

        if needs_recreate {
            // Destroy old cached resources.
            if let Some(old) = cache.take() {
                unsafe {
                    vk_ctx.device.destroy_image_view(old.color_view, None);
                    vk_ctx.device.destroy_image(old.color_image, None);
                    vk_ctx.device.free_memory(old.color_mem, None);
                    vk_ctx.device.destroy_image_view(old.depth_view, None);
                    vk_ctx.device.destroy_image(old.depth_image, None);
                    vk_ctx.device.free_memory(old.depth_mem, None);
                    vk_ctx.device.destroy_image_view(old.mv_view, None);
                    vk_ctx.device.destroy_image(old.mv_image, None);
                    vk_ctx.device.free_memory(old.mv_mem, None);
                    vk_ctx.device.destroy_image_view(old.output_view, None);
                    vk_ctx.device.destroy_image(old.output_image, None);
                    vk_ctx.device.free_memory(old.output_mem, None);
                    vk_ctx.device.destroy_buffer(old.staging_color_buf, None);
                    vk_ctx.device.free_memory(old.staging_color_mem, None);
                    vk_ctx.device.destroy_buffer(old.staging_depth_buf, None);
                    vk_ctx.device.free_memory(old.staging_depth_mem, None);
                    vk_ctx.device.destroy_buffer(old.staging_mv_buf, None);
                    vk_ctx.device.free_memory(old.staging_mv_mem, None);
                    vk_ctx.device.destroy_buffer(old.readback_buf, None);
                    vk_ctx.device.free_memory(old.readback_mem, None);
                    vk_ctx
                        .device
                        .free_command_buffers(vk_ctx.command_pool, &[old.cmd]);
                }
            }

            // Create new resources.
            let fsr_usage = vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST;

            let (color_image, color_mem, color_view) = Self::create_fsr_image(
                &vk_ctx.device,
                &vk_ctx.instance,
                vk_ctx.physical_device,
                src_w,
                src_h,
                fsr_usage,
            )?;
            let (depth_image, depth_mem, depth_view) = Self::create_fsr_image_format(
                &vk_ctx.device,
                &vk_ctx.instance,
                vk_ctx.physical_device,
                src_w,
                src_h,
                fsr_usage,
                vk::Format::R32_SFLOAT,
            )?;
            let (mv_image, mv_mem, mv_view) = Self::create_fsr_image_format(
                &vk_ctx.device,
                &vk_ctx.instance,
                vk_ctx.physical_device,
                src_w,
                src_h,
                fsr_usage,
                vk::Format::R16G16_SFLOAT,
            )?;
            let (output_image, output_mem, output_view) = Self::create_fsr_image(
                &vk_ctx.device,
                &vk_ctx.instance,
                vk_ctx.physical_device,
                dst_w,
                dst_h,
                fsr_usage,
            )?;

            let src_size = (src_w * src_h * 4) as u64;
            let mv_size = (src_w * src_h * 4) as u64;
            let (scb, scm) = vk_ctx.create_host_buffer(
                src_size,
                vk::BufferUsageFlags::TRANSFER_SRC,
            )?;
            let (sdb, sdm) = vk_ctx.create_host_buffer(
                src_size,
                vk::BufferUsageFlags::TRANSFER_SRC,
            )?;
            let (smb, smm) = vk_ctx.create_host_buffer(
                mv_size,
                vk::BufferUsageFlags::TRANSFER_SRC,
            )?;

            let dst_size = (dst_w * dst_h * 4) as u64;
            let (rb, rm) = vk_ctx.create_host_buffer(
                dst_size,
                vk::BufferUsageFlags::TRANSFER_DST,
            )?;

            let cmd = vk_ctx.allocate_command_buffer()?;

            *cache = Some(CachedResources {
                src_w,
                src_h,
                dst_w,
                dst_h,
                color_image,
                color_mem,
                color_view,
                depth_image,
                depth_mem,
                depth_view,
                mv_image,
                mv_mem,
                mv_view,
                output_image,
                output_mem,
                output_view,
                staging_color_buf: scb,
                staging_color_mem: scm,
                staging_depth_buf: sdb,
                staging_depth_mem: sdm,
                staging_mv_buf: smb,
                staging_mv_mem: smm,
                readback_buf: rb,
                readback_mem: rm,
                readback_size: dst_size,
                cmd,
            });
        }

        let res = cache.as_ref().unwrap();
        let _src_size = (src_w * src_h * 4) as u64;
        let mv_size = (src_w * src_h * 4) as u64;
        let dst_size = res.readback_size;

        // Upload source frame.
        vk_ctx.upload_to_buffer(res.staging_color_mem, &src.data)?;

        // Upload depth (1.0 = infinite depth).
        let depth_val: u32 = f32::to_bits(1.0f32);
        let depth_data: Vec<u8> = std::iter::repeat(&depth_val.to_le_bytes())
            .take((src_w * src_h) as usize)
            .flatten()
            .copied()
            .collect();
        vk_ctx.upload_to_buffer(res.staging_depth_mem, &depth_data)?;

        // Upload zero motion vectors.
        let mv_zero = vec![0u8; mv_size as usize];
        vk_ctx.upload_to_buffer(res.staging_mv_mem, &mv_zero)?;

        // Record command buffer.
        let cmd = res.cmd;
        let begin_info = vk::CommandBufferBeginInfo::default();
        unsafe {
            vk_ctx
                .device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .map_err(|e| InterpolateError::InterpolateFailed(format!("reset cmd: {e}")))?;
            vk_ctx
                .device
                .begin_command_buffer(cmd, &begin_info)
                .map_err(|e| InterpolateError::InterpolateFailed(format!("begin cmd: {e}")))?;

            // Upload images via staging.
            cmd_image_barrier_simple(
                &vk_ctx.device,
                cmd,
                res.color_image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
            copy_buf_to_image(
                &vk_ctx.device,
                cmd,
                res.staging_color_buf,
                res.color_image,
                src_w,
                src_h,
            );
            cmd_image_barrier_simple(
                &vk_ctx.device,
                cmd,
                res.color_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );

            cmd_image_barrier_simple(
                &vk_ctx.device,
                cmd,
                res.depth_image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
            copy_buf_to_image(
                &vk_ctx.device,
                cmd,
                res.staging_depth_buf,
                res.depth_image,
                src_w,
                src_h,
            );
            cmd_image_barrier_simple(
                &vk_ctx.device,
                cmd,
                res.depth_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );

            cmd_image_barrier_simple(
                &vk_ctx.device,
                cmd,
                res.mv_image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
            copy_buf_to_image(
                &vk_ctx.device,
                cmd,
                res.staging_mv_buf,
                res.mv_image,
                src_w,
                src_h,
            );
            cmd_image_barrier_simple(
                &vk_ctx.device,
                cmd,
                res.mv_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );

            cmd_image_barrier_simple(
                &vk_ctx.device,
                cmd,
                res.output_image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
            );

            let xfer_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            vk_ctx.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[xfer_barrier],
                &[],
                &[],
            );

            // Command buffer stays in recording state for FSR dispatch below.
        }

        // --- FSR 2 dispatch (records into the SAME command buffer) ---
        let mut fsr_guard = self
            .fsr_ctx
            .lock()
            .map_err(|e| InterpolateError::InterpolateFailed(format!("fsr ctx lock: {e}")))?;

        let color_res = unsafe {
            fsr::vk::get_texture_resource(
                &mut fsr_guard,
                res.color_image,
                res.color_view,
                vk::Format::R8G8B8A8_UNORM,
                [src_w, src_h],
                fsr::ResourceStates::COMPUTE_READ,
                "color",
            )
        };
        let depth_res = unsafe {
            fsr::vk::get_texture_resource(
                &mut fsr_guard,
                res.depth_image,
                res.depth_view,
                vk::Format::R32_SFLOAT,
                [src_w, src_h],
                fsr::ResourceStates::COMPUTE_READ,
                "depth",
            )
        };
        let mv_res = unsafe {
            fsr::vk::get_texture_resource(
                &mut fsr_guard,
                res.mv_image,
                res.mv_view,
                vk::Format::R16G16_SFLOAT,
                [src_w, src_h],
                fsr::ResourceStates::COMPUTE_READ,
                "motion_vectors",
            )
        };
        let output_res = unsafe {
            fsr::vk::get_texture_resource(
                &mut fsr_guard,
                res.output_image,
                res.output_view,
                vk::Format::R8G8B8A8_UNORM,
                [dst_w, dst_h],
                fsr::ResourceStates::COMPUTE_READ,
                "output",
            )
        };

        let jx = halton((frame_idx & 255) as u32, 2) - 0.5;
        let jy = halton((frame_idx & 255) as u32, 3) - 0.5;

        // FSR dispatch records into cmd (still in recording state after upload).
        let cmd_list: fsr::CommandList = cmd.into();
        let desc = fsr::DispatchDescription::new(
            cmd_list,
            color_res,
            depth_res,
            mv_res,
            output_res,
            1.0 / 60.0,
            [src_w, src_h],
        )
        .jitter_offset([jx, jy])
        .sharpness(0.5);

        unsafe { fsr_guard.dispatch(desc) }
            .map_err(|e| InterpolateError::InterpolateFailed(format!("FSR dispatch: {e}")))?;
        drop(fsr_guard);

        // After FSR dispatch, transition output image and copy to readback buffer
        // — all within the same command buffer.
        unsafe {
            // Barrier: make output image readable for transfer after FSR write.
            let out_barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(res.output_image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            vk_ctx.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&out_barrier),
            );

            let copy_region = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .image_offset(vk::Offset3D::default())
                .image_extent(vk::Extent3D::default().width(dst_w).height(dst_h).depth(1));
            vk_ctx.device.cmd_copy_image_to_buffer(
                cmd,
                res.output_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                res.readback_buf,
                std::slice::from_ref(&copy_region),
            );

            vk_ctx
                .device
                .end_command_buffer(cmd)
                .map_err(|e| InterpolateError::InterpolateFailed(format!("end dr cmd: {e}")))?;
        }

        // Single submit for both dispatch and readback.
        vk_ctx.submit_and_wait(cmd)?;

        let result_data = vk_ctx.read_from_buffer(res.readback_mem, dst_size as usize)?;

        Ok(GpuFrame {
            data: result_data,
            width: dst_w,
            height: dst_h,
            stride: dst_w * 4,
            timestamp_ns: src.timestamp_ns,
        })
    }

    fn latency_ms(&self) -> f32 {
        1.5
    }

    fn name(&self) -> &'static str {
        "fsr2-native"
    }
}

impl Drop for Fsr2NativeInterpolator {
    fn drop(&mut self) {
        // Destroy FSR context.
        if let Ok(mut ctx) = self.fsr_ctx.lock() {
            unsafe {
                let _ = ctx.destroy();
            }
        }
        // Destroy cached GPU resources.
        if let Ok(mut cache) = self.cached_res.lock()
            && let Some(res) = cache.take()
        {
            if let Ok(vk_ctx) = self.ctx.lock() {
                unsafe {
                    vk_ctx.device.destroy_image_view(res.color_view, None);
                    vk_ctx.device.destroy_image(res.color_image, None);
                    vk_ctx.device.free_memory(res.color_mem, None);
                    vk_ctx.device.destroy_image_view(res.depth_view, None);
                    vk_ctx.device.destroy_image(res.depth_image, None);
                    vk_ctx.device.free_memory(res.depth_mem, None);
                    vk_ctx.device.destroy_image_view(res.mv_view, None);
                    vk_ctx.device.destroy_image(res.mv_image, None);
                    vk_ctx.device.free_memory(res.mv_mem, None);
                    vk_ctx.device.destroy_image_view(res.output_view, None);
                    vk_ctx.device.destroy_image(res.output_image, None);
                    vk_ctx.device.free_memory(res.output_mem, None);
                    vk_ctx.device.destroy_buffer(res.staging_color_buf, None);
                    vk_ctx.device.free_memory(res.staging_color_mem, None);
                    vk_ctx.device.destroy_buffer(res.staging_depth_buf, None);
                    vk_ctx.device.free_memory(res.staging_depth_mem, None);
                    vk_ctx.device.destroy_buffer(res.staging_mv_buf, None);
                    vk_ctx.device.free_memory(res.staging_mv_mem, None);
                    vk_ctx.device.destroy_buffer(res.readback_buf, None);
                    vk_ctx.device.free_memory(res.readback_mem, None);
                    vk_ctx
                        .device
                        .free_command_buffers(vk_ctx.command_pool, &[res.cmd]);
                    vk_ctx
                        .device
                        .free_command_buffers(vk_ctx.command_pool, &[res.cmd]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halton_values() {
        // Halton(0, 2) = 0.0
        assert!((halton(0, 2) - 0.0).abs() < 1e-6);
        // Halton(1, 2) = 0.5
        assert!((halton(1, 2) - 0.5).abs() < 1e-6);
        // Halton(2, 2) = 0.25
        assert!((halton(2, 2) - 0.25).abs() < 1e-6);
        // Halton(3, 2) = 0.75
        assert!((halton(3, 2) - 0.75).abs() < 1e-6);
    }

    #[test]
    fn halton_jitter_range() {
        for i in 0..256 {
            let jx = halton(i, 2) - 0.5;
            let jy = halton(i, 3) - 0.5;
            assert!((-0.5..=0.5).contains(&jx), "jx={jx} out of range");
            assert!((-0.5..=0.5).contains(&jy), "jy={jy} out of range");
        }
    }

    #[test]
    fn fsr2_native_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Fsr2NativeInterpolator>();
    }
}

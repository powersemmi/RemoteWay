//! AMD FSR 2 native upscaling backend via the Embark Studios `fsr` crate.
//!
//! Uses the real [`fsr::Context`] to perform spatial upscaling. Motion vectors
//! are passed as zero and the temporal history is reset every frame, because
//! a screen-capture stream has no engine-generated jitter / MVs to drive
//! FSR2's TAA-style accumulator — feeding it stale history produces ghosting.
//! Temporal interpolation (frame generation) is not supported by this
//! backend — use [`super::fsr3`] for that.

#![allow(clippy::undocumented_unsafe_blocks)]

use std::sync::{Arc, Mutex};

use ash::vk;

use crate::error::InterpolateError;
use crate::interpolator::{FrameInterpolator, GpuFrame};

use super::vulkan_context::VulkanContext;

/// Cached per-resolution Vulkan resources for FSR 2 upscaling.
///
/// Images, buffers and the FSR context itself are kept alive across frames
/// and recreated only when input/output dimensions change. The FSR context
/// is bound to a specific `(max_render_size, display_size)` at creation —
/// using a context whose `display_size` does not match the actual output
/// texture leads to wrong sampling ratios (the symptom: only the top-left
/// fraction of the source ends up scaled into the output).
struct CachedResources {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    /// FSR 2 context, created with `display_size = [dst_w, dst_h]` and
    /// `max_render_size = [src_w, src_h]`. Owned here so we can destroy
    /// it cleanly when dimensions change.
    fsr_ctx: fsr::Context,
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
/// The FSR context is created lazily (and recreated on every dimension
/// change) inside [`CachedResources`], because FSR2's `display_size` is
/// fixed at context creation time and must match the actual destination
/// texture for sampling to be correct.
pub struct Fsr2NativeInterpolator {
    /// Shared Vulkan device, memory, and queue.
    ctx: Arc<Mutex<VulkanContext>>,
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
        _display_w: u32,
        _display_h: u32,
        _max_render_w: u32,
        _max_render_h: u32,
    ) -> Result<Self, InterpolateError> {
        let vk_ctx = Arc::new(Mutex::new(VulkanContext::new(&[])?));
        Self::with_context(vk_ctx, _display_w, _display_h, _max_render_w, _max_render_h)
    }

    /// Create a new FSR 2 native upscaler sharing an existing [`VulkanContext`].
    ///
    /// The `display_*` / `max_render_*` arguments are accepted only for API
    /// compatibility; the actual FSR context is created lazily on the first
    /// `upscale()` call with `display_size` matching the real destination
    /// resolution (and recreated whenever it changes). Eagerly fixing the
    /// context to some "max" resolution would force every dispatch into the
    /// wrong scaling ratio and clip the output to the upper-left fraction
    /// of the source.
    pub(crate) fn with_context(
        vk_ctx: Arc<Mutex<VulkanContext>>,
        _display_w: u32,
        _display_h: u32,
        _max_render_w: u32,
        _max_render_h: u32,
    ) -> Result<Self, InterpolateError> {
        let guard = vk_ctx
            .lock()
            .map_err(|e| InterpolateError::InitFailed(format!("mutex poisoned: {e}")))?;

        // Probe the FSR Vulkan interface to fail fast if the SDK / driver
        // combo cannot host FSR at all. The interface itself is discarded —
        // it cannot be reused across contexts, and the per-resolution
        // context built later in `upscale` will recreate it.
        let _probe = unsafe {
            fsr::vk::get_interface(&guard._entry, &guard.instance, guard.physical_device)
        }
        .map_err(|e| InterpolateError::InitFailed(format!("FSR interface: {e}")))?;
        drop(_probe);
        drop(guard);

        Ok(Self {
            ctx: vk_ctx,
            cached_res: Mutex::new(None),
        })
    }

    /// Build a fresh FSR 2 context for the given input/output resolution.
    ///
    /// `display_size` must equal the output texture's `dst_w × dst_h` —
    /// FSR2 uses it directly as the dispatch grid and the sampling scale.
    fn build_fsr_context(
        vk_ctx: &VulkanContext,
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
    ) -> Result<fsr::Context, InterpolateError> {
        let interface = unsafe {
            fsr::vk::get_interface(&vk_ctx._entry, &vk_ctx.instance, vk_ctx.physical_device)
        }
        .map_err(|e| InterpolateError::InterpolateFailed(format!("FSR interface: {e}")))?;

        let fsr_device = unsafe { fsr::vk::get_device(vk_ctx.device.clone()) };

        let flags = fsr::InitializationFlagBits::ENABLE_DEPTH_INFINITE
            | fsr::InitializationFlagBits::ENABLE_DEPTH_INVERTED;

        let desc = fsr::ContextDescription {
            interface,
            flags,
            max_render_size: [src_w, src_h],
            display_size: [dst_w, dst_h],
            device: &fsr_device,
            message_callback: None,
        };

        unsafe { fsr::Context::new(desc) }
            .map_err(|e| InterpolateError::InterpolateFailed(format!("FSR context: {e}")))
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
            // Destroy old cached resources, including the FSR context whose
            // display_size is bound to the previous output resolution.
            if let Some(mut old) = cache.take() {
                unsafe {
                    let _ = old.fsr_ctx.destroy();
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

            // Build a fresh FSR context whose `display_size` matches the
            // actual output texture; this is what fixes the "only the
            // top-left fraction of the source is visible" symptom.
            let fsr_ctx = Self::build_fsr_context(&vk_ctx, src_w, src_h, dst_w, dst_h)?;

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

            // One-time GPU initialization of depth and motion-vector images:
            // depth = 1.0 (far plane under inverted-infinite depth), MVs = 0.
            // Both are constants for screen-capture upscale, so we upload
            // their pixel data + execute the buf→image copies + transition
            // to SHADER_READ_ONLY_OPTIMAL exactly once here. The per-frame
            // dispatch then skips all of this work and reads the cached
            // images directly. Saves ~7 MB CPU→GPU copy and ~6 image
            // barriers per frame.
            let depth_bytes = f32::to_bits(1.0f32).to_le_bytes();
            let pixels = (src_w * src_h) as usize;
            let mut depth_scratch = Vec::with_capacity(pixels * 4);
            for _ in 0..pixels {
                depth_scratch.extend_from_slice(&depth_bytes);
            }
            let mv_scratch = vec![0u8; mv_size as usize];
            vk_ctx.upload_to_buffer(sdm, &depth_scratch)?;
            vk_ctx.upload_to_buffer(smm, &mv_scratch)?;
            drop(depth_scratch);
            drop(mv_scratch);

            let init_cmd = vk_ctx.allocate_command_buffer()?;
            unsafe {
                vk_ctx
                    .device
                    .begin_command_buffer(init_cmd, &vk::CommandBufferBeginInfo::default())
                    .map_err(|e| {
                        InterpolateError::InterpolateFailed(format!("begin init cmd: {e}"))
                    })?;
                cmd_image_barrier_simple(
                    &vk_ctx.device,
                    init_cmd,
                    depth_image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                );
                copy_buf_to_image(&vk_ctx.device, init_cmd, sdb, depth_image, src_w, src_h);
                cmd_image_barrier_simple(
                    &vk_ctx.device,
                    init_cmd,
                    depth_image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                );
                cmd_image_barrier_simple(
                    &vk_ctx.device,
                    init_cmd,
                    mv_image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                );
                copy_buf_to_image(&vk_ctx.device, init_cmd, smb, mv_image, src_w, src_h);
                cmd_image_barrier_simple(
                    &vk_ctx.device,
                    init_cmd,
                    mv_image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                );
                vk_ctx
                    .device
                    .end_command_buffer(init_cmd)
                    .map_err(|e| InterpolateError::InterpolateFailed(format!("end init cmd: {e}")))?;
            }
            vk_ctx.submit_and_wait(init_cmd)?;
            unsafe {
                vk_ctx
                    .device
                    .free_command_buffers(vk_ctx.command_pool, &[init_cmd]);
            }

            *cache = Some(CachedResources {
                src_w,
                src_h,
                dst_w,
                dst_h,
                fsr_ctx,
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

        // Mutable borrow — needed because `fsr::Context::dispatch` and
        // `fsr::vk::get_texture_resource` both take `&mut fsr::Context`.
        let res = cache.as_mut().unwrap();
        let dst_size = res.readback_size;

        // Upload only the color frame — depth and motion-vector images
        // were filled once when the cache was built and stay in
        // SHADER_READ_ONLY_OPTIMAL across frames.
        let upload_t0 = std::time::Instant::now();
        vk_ctx.upload_to_buffer(res.staging_color_mem, &src.data)?;
        let upload_ms = upload_t0.elapsed().as_secs_f32() * 1000.0;

        // Record command buffer. Only the color image and output image are
        // touched here; the depth/MV images stay in SHADER_READ from init.
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

            // Color image: contents are fully overwritten each frame, so
            // UNDEFINED as the source layout is fine (discards previous
            // contents). After the copy we transition to SHADER_READ so
            // FSR2 can sample it.
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

            // Output image: same UNDEFINED→GENERAL trick. FSR2 will write
            // every pixel during dispatch.
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
        // The FSR context lives in the cache (one per `(src, dst)`) — no
        // separate mutex to acquire.
        let color_res = unsafe {
            fsr::vk::get_texture_resource(
                &mut res.fsr_ctx,
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
                &mut res.fsr_ctx,
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
                &mut res.fsr_ctx,
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
                &mut res.fsr_ctx,
                res.output_image,
                res.output_view,
                vk::Format::R8G8B8A8_UNORM,
                [dst_w, dst_h],
                fsr::ResourceStates::COMPUTE_READ,
                "output",
            )
        };

        // Screen-capture frames carry no sub-pixel jitter and no engine
        // motion vectors — feeding FSR2 stale jitter offsets + zero MVs
        // builds up a temporal history that ghosts everything that moves
        // between frames. We pin jitter to (0,0) (so FSR doesn't expect
        // off-grid sampling) and pass `reset=true` every dispatch so the
        // history buffer is discarded each frame.
        //
        // With history disabled, FSR2's spatial reconstruction kernel
        // alone tends to soften high-frequency edges (text strokes, UI
        // lines). We compensate by enabling FSR2's internal RCAS pass at
        // a strong setting — `sharpness` >= 0.8 keeps glyphs crisp on the
        // upscaled image while still being clamped by FSR2's own
        // ringing limiter, so we get sharp text without halos.
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
        .jitter_offset([0.0, 0.0])
        .reset(true)
        .sharpness(0.85);

        unsafe { res.fsr_ctx.dispatch(desc) }
            .map_err(|e| InterpolateError::InterpolateFailed(format!("FSR dispatch: {e}")))?;

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

        // Single submit for both dispatch and readback. This blocks the
        // calling thread until the GPU finishes, which is the dominant
        // cost of the FSR2 upscale path — log wall-time so the user can
        // see whether it's actually the bottleneck.
        let submit_t0 = std::time::Instant::now();
        vk_ctx.submit_and_wait(cmd)?;
        let submit_ms = submit_t0.elapsed().as_secs_f32() * 1000.0;

        let readback_t0 = std::time::Instant::now();
        let result_data = vk_ctx.read_from_buffer(res.readback_mem, dst_size as usize)?;
        let readback_ms = readback_t0.elapsed().as_secs_f32() * 1000.0;

        tracing::debug!(
            src = format!("{src_w}x{src_h}"),
            dst = format!("{dst_w}x{dst_h}"),
            upload_ms,
            gpu_ms = submit_ms,
            readback_ms,
            total_ms = upload_ms + submit_ms + readback_ms,
            "fsr2 upscale done"
        );

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
        if let Ok(mut cache) = self.cached_res.lock()
            && let Some(mut res) = cache.take()
            && let Ok(vk_ctx) = self.ctx.lock()
        {
            unsafe {
                let _ = res.fsr_ctx.destroy();
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
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fsr2_native_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Fsr2NativeInterpolator>();
    }
}

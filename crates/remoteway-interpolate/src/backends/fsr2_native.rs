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

/// Block size for CPU motion estimation.
const BLOCK_SIZE: u32 = 16;

/// Search radius in pixels for block matching.
const SEARCH_RADIUS: u32 = 8;

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

/// Real AMD FSR 2 upscaling via the `fsr` crate (Vulkan backend).
///
/// Holds an [`fsr::Context`] for FSR 2 dispatches and a shared
/// [`VulkanContext`] for buffer/image management. Motion vectors are
/// computed on the CPU with simple block-matching; when that fails,
/// zero vectors are used (which disables temporal accumulation but
/// still produces a spatially upscaled result).
pub struct Fsr2NativeInterpolator {
    /// The FSR 2 context (interior-mutable via Mutex because
    /// `dispatch()` requires `&mut self`).
    fsr_ctx: Mutex<fsr::Context>,
    /// Shared Vulkan device, memory, and queue.
    ctx: Arc<Mutex<VulkanContext>>,
    /// Previous frame for motion estimation.
    prev_frame: Mutex<Option<GpuFrame>>,
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
}

/// SAFETY: All GPU resources are guarded by `Arc<Mutex<>>` and
/// `fsr::Context` dispatch is serialized through its own Mutex.
// SAFETY: All mutable state is protected by internal synchronization (Mutex).
unsafe impl Send for Fsr2NativeInterpolator {}
// SAFETY: All mutable state is protected by internal synchronization (Mutex).
unsafe impl Sync for Fsr2NativeInterpolator {}

impl Fsr2NativeInterpolator {
    /// Create a new FSR 2 native upscaler.
    ///
    /// # Parameters
    /// - `display_w`, `display_h` — output (display) resolution.
    /// - `max_render_w`, `max_render_h` — maximum input (render) resolution.
    ///
    /// # Errors
    /// Returns [`InterpolateError::InitFailed`] if Vulkan is unavailable or
    /// the FSR 2 context cannot be created.
    pub fn new(
        display_w: u32,
        display_h: u32,
        max_render_w: u32,
        max_render_h: u32,
    ) -> Result<Self, InterpolateError> {
        let vk_ctx = Arc::new(Mutex::new(VulkanContext::new(&[])?));

        // Create the FSR 2 interface and context.
        let guard = vk_ctx
            .lock()
            .map_err(|e| InterpolateError::InitFailed(format!("mutex poisoned: {e}")))?;

        // SAFETY: The ash Entry, Instance, and PhysicalDevice are all valid
        // handles from VulkanContext::new(). get_interface reads the device
        // extension properties and creates an internal scratch buffer.
        let interface = unsafe {
            fsr::vk::get_interface(&guard._entry, &guard.instance, guard.physical_device)
        }
        .map_err(|e| InterpolateError::InitFailed(format!("FSR interface: {e}")))?;

        // SAFETY: The ash Device handle is valid. get_device wraps it for FSR.
        let fsr_device = unsafe { fsr::vk::get_device(guard.device.clone()) };

        let flags = fsr::InitializationFlagBits::ENABLE_HIGH_DYNAMIC_RANGE
            | fsr::InitializationFlagBits::ENABLE_DEPTH_INFINITE
            | fsr::InitializationFlagBits::ENABLE_DEPTH_INVERTED;

        let context_desc = fsr::ContextDescription {
            interface,
            flags,
            max_render_size: [max_render_w, max_render_h],
            display_size: [display_w, display_h],
            device: &fsr_device,
            message_callback: None,
        };

        // SAFETY: context_desc references valid Vulkan handles and the
        // interface was created above. The scratch buffer backing the
        // interface outlives the context because Interface owns it.
        let fsr_ctx = unsafe { fsr::Context::new(context_desc) }
            .map_err(|e| InterpolateError::InitFailed(format!("FSR context: {e}")))?;

        drop(guard);

        Ok(Self {
            fsr_ctx: Mutex::new(fsr_ctx),
            ctx: vk_ctx,
            prev_frame: Mutex::new(None),
            frame_idx: Mutex::new(0),
            display_w,
            display_h,
            max_render_w,
            max_render_h,
        })
    }

    /// Compute per-block motion vectors between `prev` and `curr` on the CPU.
    ///
    /// Divides the frame into [`BLOCK_SIZE`]×[`BLOCK_SIZE`] blocks and searches
    /// within ±[`SEARCH_RADIUS`] pixels for the best SAD match. Returns
    /// `(dx, dy)` pairs packed as `[dx0, dy0, dx1, dy1, …]` in RGBA8
    /// (two `i16` per `[u8; 4]` pixel, suitable for the motion-vector texture).
    #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    fn compute_motion_vectors(prev: &GpuFrame, curr: &GpuFrame) -> Vec<u8> {
        let w = curr.width;
        let h = curr.height;
        let stride = curr.stride;

        // Motion vector image: 2 components (x, y) as f16 in RG format,
        // but we store as raw i16 pairs packed into [u8; 4] per "pixel".
        // The FSR2 motion vector texture is render-resolution with
        // R16G16_SFLOAT format. For simplicity we use R8G8B8A8_UNORM
        // and store the vectors as normalized (0.5 + v / max_range).
        let mut mv_data = vec![128u8; (w * h * 4) as usize]; // neutral grey = zero motion

        let block_w = w.div_ceil(BLOCK_SIZE);
        let block_h = h.div_ceil(BLOCK_SIZE);
        let max_disp = SEARCH_RADIUS as f32;

        for by in 0..block_h {
            let y0 = by * BLOCK_SIZE;
            let y1 = (y0 + BLOCK_SIZE).min(h);
            for bx in 0..block_w {
                let x0 = bx * BLOCK_SIZE;
                let x1 = (x0 + BLOCK_SIZE).min(w);

                let (best_dx, best_dy) = Self::block_match(prev, curr, stride, x0, y0, x1, y1);

                // Normalize to [0, 1] for UNORM storage: 0.5 = zero
                let nx = (best_dx as f32 / max_disp * 0.5 + 0.5).clamp(0.0, 1.0);
                let ny = (best_dy as f32 / max_disp * 0.5 + 0.5).clamp(0.0, 1.0);
                let r = (nx * 255.0) as u8;
                let g = (ny * 255.0) as u8;

                // Fill the block with the computed vector.
                for py in y0..y1 {
                    let row_off = (py * w * 4) as usize;
                    for px in x0..x1 {
                        let off = row_off + (px * 4) as usize;
                        mv_data[off] = r;
                        mv_data[off + 1] = g;
                        // B and A channels stay at 128 (neutral).
                    }
                }
            }
        }

        mv_data
    }

    /// Simple block-match: find the best offset within ±`SEARCH_RADIUS`.
    #[allow(clippy::too_many_arguments)]
    fn block_match(
        prev: &GpuFrame,
        curr: &GpuFrame,
        stride: u32,
        x0: u32,
        y0: u32,
        x1: u32,
        y1: u32,
    ) -> (i32, i32) {
        let w = curr.width as i32;
        let h = curr.height as i32;

        let sr = SEARCH_RADIUS as i32;
        let mut best = (0i32, 0i32);
        let mut best_sad = u64::MAX;
        for dy in -sr..=sr {
            for dx in -sr..=sr {
                let mut sad = 0u64;
                'block: for py in y0..y1 {
                    let sy = py as i32 + dy;
                    if sy < 0 || sy >= h {
                        sad = u64::MAX;
                        break 'block;
                    }
                    let prev_row = (sy as u32 * prev.stride) as usize;
                    let curr_row = (py * stride) as usize;
                    for px in x0..x1 {
                        let sx = px as i32 + dx;
                        if sx < 0 || sx >= w {
                            sad = u64::MAX;
                            break 'block;
                        }
                        let pi = prev_row + (sx as u32 * 4) as usize;
                        let ci = curr_row + (px * 4) as usize;
                        let pl = u16::from(prev.data[pi]);
                        let cl = u16::from(curr.data[ci]);
                        sad += if pl > cl {
                            (pl - cl) as u64
                        } else {
                            (cl - pl) as u64
                        };
                        if sad >= best_sad {
                            break 'block;
                        }
                    }
                }
                if sad < best_sad {
                    best_sad = sad;
                    best = (dx, dy);
                }
            }
        }

        best
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

    /// Upload RGBA8 data to a device-local image via a staging buffer.
    #[allow(clippy::too_many_arguments)]
    fn upload_to_image(
        vk_ctx: &VulkanContext,
        cmd: vk::CommandBuffer,
        image: vk::Image,
        data: &[u8],
        width: u32,
        height: u32,
    ) -> Result<(), InterpolateError> {
        let size = (width * height * 4) as u64;
        let (staging_buf, staging_mem) = vk_ctx.create_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        vk_ctx.upload_to_buffer(staging_mem, data)?;

        // Transition image to TRANSFER_DST_OPTIMAL.
        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
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
            vk_ctx.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&barrier),
            );
        }

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
            .image_extent(vk::Extent3D::default().width(width).height(height).depth(1));
        unsafe {
            vk_ctx.device.cmd_copy_buffer_to_image(
                cmd,
                staging_buf,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&copy_region),
            );
        }

        // Transition to SHADER_READ_ONLY_OPTIMAL for FSR 2 sampling.
        let barrier2 = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
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
            vk_ctx.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&barrier2),
            );
        }

        // Cleanup staging buffer after submit.
        unsafe {
            vk_ctx.device.destroy_buffer(staging_buf, None);
            vk_ctx.device.free_memory(staging_mem, None);
        }

        Ok(())
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
    /// AMD FSR 2 with temporal feedback.
    ///
    /// # Errors
    /// Returns [`InterpolateError::InterpolateFailed`] if any Vulkan or FSR 2
    /// operation fails.
    #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    fn upscale(
        &self,
        src: &GpuFrame,
        dst_w: u32,
        dst_h: u32,
    ) -> Result<GpuFrame, InterpolateError> {
        let src_w = src.width;
        let src_h = src.height;

        // Acquire frame index and increment.
        let mut fidx = self
            .frame_idx
            .lock()
            .map_err(|e| InterpolateError::InterpolateFailed(format!("frame_idx lock: {e}")))?;
        let frame_idx = *fidx;
        *fidx = frame_idx.wrapping_add(1);
        drop(fidx);

        // Compute motion vectors (CPU block-match or zero).
        let motion_data = {
            let prev = self.prev_frame.lock().map_err(|e| {
                InterpolateError::InterpolateFailed(format!("prev_frame lock: {e}"))
            })?;
            if let Some(ref prev_frame) = *prev {
                if prev_frame.same_dimensions(src) {
                    Self::compute_motion_vectors(prev_frame, src)
                } else {
                    // Dimension change: zero vectors.
                    vec![128u8; (src_w * src_h * 4) as usize]
                }
            } else {
                // No previous frame: zero vectors.
                vec![128u8; (src_w * src_h * 4) as usize]
            }
        };

        // Store current frame as previous for next call.
        {
            let mut prev = self.prev_frame.lock().map_err(|e| {
                InterpolateError::InterpolateFailed(format!("prev_frame lock: {e}"))
            })?;
            *prev = Some(GpuFrame::from_data(
                src.data.clone(),
                src.width,
                src.height,
                src.stride,
                src.timestamp_ns,
            ));
        }

        // Lock Vulkan context.
        let vk_ctx = self
            .ctx
            .lock()
            .map_err(|e| InterpolateError::InterpolateFailed(format!("vk ctx lock: {e}")))?;

        // Image usage for FSR 2 resources.
        let fsr_usage = vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST;

        // Create GPU images.
        let (color_image, color_mem, color_view) = Self::create_fsr_image(
            &vk_ctx.device,
            &vk_ctx.instance,
            vk_ctx.physical_device,
            src_w,
            src_h,
            fsr_usage,
        )?;
        let (depth_image, depth_mem, depth_view) = Self::create_fsr_image(
            &vk_ctx.device,
            &vk_ctx.instance,
            vk_ctx.physical_device,
            src_w,
            src_h,
            fsr_usage,
        )?;
        let (mv_image, mv_mem, mv_view) = Self::create_fsr_image(
            &vk_ctx.device,
            &vk_ctx.instance,
            vk_ctx.physical_device,
            src_w,
            src_h,
            fsr_usage,
        )?;
        let (output_image, output_mem, output_view) = Self::create_fsr_image(
            &vk_ctx.device,
            &vk_ctx.instance,
            vk_ctx.physical_device,
            dst_w,
            dst_h,
            fsr_usage,
        )?;

        // Create readback buffer.
        let dst_size = (dst_w * dst_h * 4) as u64;
        let (readback_buf, readback_mem) = vk_ctx.create_buffer(
            dst_size,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        // Allocate command buffer and begin recording.
        let cmd = vk_ctx.allocate_command_buffer()?;
        let begin_info = vk::CommandBufferBeginInfo::default();
        unsafe { vk_ctx.device.begin_command_buffer(cmd, &begin_info) }
            .map_err(|e| InterpolateError::InterpolateFailed(format!("begin cmd: {e}")))?;

        // Upload color, depth (1.0), and motion vectors.
        Self::upload_to_image(&vk_ctx, cmd, color_image, &src.data, src_w, src_h)?;

        // Depth buffer: all 1.0 (max depth for DEPTH_INFINITE + DEPTH_INVERTED).
        let depth_data = vec![255u8; (src_w * src_h * 4) as usize];
        Self::upload_to_image(&vk_ctx, cmd, depth_image, &depth_data, src_w, src_h)?;

        // Motion vectors.
        Self::upload_to_image(&vk_ctx, cmd, mv_image, &motion_data, src_w, src_h)?;

        // Transition output image for FSR 2 write.
        let out_barrier = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(output_image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );
        unsafe {
            vk_ctx.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&out_barrier),
            );
        }

        // End command buffer and submit (upload + barriers must complete
        // before FSR dispatch accesses the images).
        unsafe { vk_ctx.device.end_command_buffer(cmd) }
            .map_err(|e| InterpolateError::InterpolateFailed(format!("end cmd: {e}")))?;
        vk_ctx.submit_and_wait(cmd)?;

        // --- FSR 2 dispatch ---
        let mut fsr_guard = self
            .fsr_ctx
            .lock()
            .map_err(|e| InterpolateError::InterpolateFailed(format!("fsr ctx lock: {e}")))?;

        // Create FSR resources.
        // SAFETY: All images and views are valid Vulkan handles created above.
        // The FSR context is valid (created in new()). Names are ASCII.
        let color_res = unsafe {
            fsr::vk::get_texture_resource(
                &mut fsr_guard,
                color_image,
                color_view,
                vk::Format::R8G8B8A8_UNORM,
                [src_w, src_h],
                fsr::ResourceStates::COMPUTE_READ,
                "color",
            )
        };
        let depth_res = unsafe {
            fsr::vk::get_texture_resource(
                &mut fsr_guard,
                depth_image,
                depth_view,
                vk::Format::R8G8B8A8_UNORM,
                [src_w, src_h],
                fsr::ResourceStates::COMPUTE_READ,
                "depth",
            )
        };
        let mv_res = unsafe {
            fsr::vk::get_texture_resource(
                &mut fsr_guard,
                mv_image,
                mv_view,
                vk::Format::R8G8B8A8_UNORM,
                [src_w, src_h],
                fsr::ResourceStates::COMPUTE_READ,
                "motion_vectors",
            )
        };
        let output_res = unsafe {
            fsr::vk::get_texture_resource(
                &mut fsr_guard,
                output_image,
                output_view,
                vk::Format::R8G8B8A8_UNORM,
                [dst_w, dst_h],
                fsr::ResourceStates::COMPUTE_READ,
                "output",
            )
        };

        // Compute Halton jitter.
        let jx = halton((frame_idx & 255) as u32, 2) - 0.5;
        let jy = halton((frame_idx & 255) as u32, 3) - 0.5;

        // Allocate a fresh command buffer for the FSR dispatch.
        let dispatch_cmd = vk_ctx.allocate_command_buffer()?;
        let begin_info = vk::CommandBufferBeginInfo::default();
        unsafe {
            vk_ctx
                .device
                .begin_command_buffer(dispatch_cmd, &begin_info)
        }
        .map_err(|e| InterpolateError::InterpolateFailed(format!("begin dispatch cmd: {e}")))?;

        let cmd_list: fsr::CommandList = dispatch_cmd.into();

        let desc = fsr::DispatchDescription::new(
            cmd_list,
            color_res,
            depth_res,
            mv_res,
            output_res,
            1.0 / 60.0, // frame_time_delta: assume 60 FPS
            [src_w, src_h],
        )
        .jitter_offset([jx, jy])
        .sharpness(0.5);

        // SAFETY: All resources are valid, the command buffer is recording,
        // and the FSR context was created with matching dimensions.
        unsafe { fsr_guard.dispatch(desc) }
            .map_err(|e| InterpolateError::InterpolateFailed(format!("FSR dispatch: {e}")))?;

        drop(fsr_guard);

        // End and submit the dispatch command buffer.
        unsafe { vk_ctx.device.end_command_buffer(dispatch_cmd) }
            .map_err(|e| InterpolateError::InterpolateFailed(format!("end dispatch cmd: {e}")))?;
        vk_ctx.submit_and_wait(dispatch_cmd)?;

        // --- Read back output image ---
        let readback_cmd = vk_ctx.allocate_command_buffer()?;
        let begin_info = vk::CommandBufferBeginInfo::default();
        unsafe {
            vk_ctx
                .device
                .begin_command_buffer(readback_cmd, &begin_info)
        }
        .map_err(|e| InterpolateError::InterpolateFailed(format!("begin readback cmd: {e}")))?;

        // Transition output image for copy.
        let copy_barrier = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(output_image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );
        unsafe {
            vk_ctx.device.cmd_pipeline_barrier(
                readback_cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&copy_barrier),
            );
        }

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
        unsafe {
            vk_ctx.device.cmd_copy_image_to_buffer(
                readback_cmd,
                output_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                readback_buf,
                std::slice::from_ref(&copy_region),
            );
        }

        unsafe { vk_ctx.device.end_command_buffer(readback_cmd) }
            .map_err(|e| InterpolateError::InterpolateFailed(format!("end readback cmd: {e}")))?;
        vk_ctx.submit_and_wait(readback_cmd)?;

        let result_data = vk_ctx.read_from_buffer(readback_mem, dst_size as usize)?;

        // --- Cleanup ---
        unsafe {
            vk_ctx.device.destroy_image_view(color_view, None);
            vk_ctx.device.destroy_image(color_image, None);
            vk_ctx.device.free_memory(color_mem, None);
            vk_ctx.device.destroy_image_view(depth_view, None);
            vk_ctx.device.destroy_image(depth_image, None);
            vk_ctx.device.free_memory(depth_mem, None);
            vk_ctx.device.destroy_image_view(mv_view, None);
            vk_ctx.device.destroy_image(mv_image, None);
            vk_ctx.device.free_memory(mv_mem, None);
            vk_ctx.device.destroy_image_view(output_view, None);
            vk_ctx.device.destroy_image(output_image, None);
            vk_ctx.device.free_memory(output_mem, None);
            vk_ctx.device.destroy_buffer(readback_buf, None);
            vk_ctx.device.free_memory(readback_mem, None);
            // Free the two command buffers.
            vk_ctx
                .device
                .free_command_buffers(vk_ctx.command_pool, &[cmd]);
            vk_ctx
                .device
                .free_command_buffers(vk_ctx.command_pool, &[dispatch_cmd, readback_cmd]);
        }

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
        // SAFETY: The FSR context is valid and no dispatches are in flight.
        // The VulkanContext is dropped separately via Arc.
        if let Ok(mut ctx) = self.fsr_ctx.lock() {
            unsafe {
                let _ = ctx.destroy();
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

//! AMD FSR3 Frame Generation backend for Linux/Wayland (Vulkan).
//!
//! Uses the `FidelityFX` SDK (`libFidelityFX.so`) compiled at build time via `build.rs` (static link).
//! Requires SDK v1.1.x built with Vulkan support.
//!
//! ## Pipeline
//!
//! ```text
//! Init:
//!   ffxGetInterfaceVK()          → Vulkan backend function table
//!   ffxFrameInterpolationContextCreate() → FG context
//!   ffxFrameInterpolationGetSharedResourceDescriptions() → shared resources
//!
//! Per frame:
//!   upload frame → VkImage
//!   ffxFrameInterpolationPrepare()  → depth + motion vectors + jitter
//!   ffxFrameInterpolationDispatch() → generate interpolated frame
//!   read back → GpuFrame
//! ```
//!
//! ## Setup
//!
//! Build the `FidelityFX` SDK and set `FIDELITYFX_LIB_PATH`:
//!
//! ```bash
//! git clone https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK
//! cd FidelityFX-SDK && git checkout v1.1.4 && cd sdk
//! cmake -B build -DCMAKE_BUILD_TYPE=Release
//! cmake --build build
//! export FIDELITYFX_LIB_PATH=$PWD/build/bin/libFidelityFX.so
//! ```
#![allow(clippy::undocumented_unsafe_blocks)]
use std::sync::{Arc, Mutex};
use ash::vk;
use ash::vk::Handle;
use crate::error::InterpolateError;
use crate::interpolator::{FrameInterpolator, GpuFrame};
use super::ffx_fg::{
    FFX_BACKBUFFER_TRANSFER_FUNCTION_SRGB, FFX_FRAMEINTERPOLATION_DISPATCH_DRAW_DEBUG_TEAR_LINES,
    FFX_FRAMEINTERPOLATION_ENABLE_DEPTH_INFINITE, FFX_FRAMEINTERPOLATION_ENABLE_DEPTH_INVERTED,
    FFX_FRAMEINTERPOLATION_ENABLE_HDR_COLOR_INPUT,
    FFX_FRAMEINTERPOLATION_ENABLE_JITTER_MOTION_VECTORS, FFX_OK, FFX_RESOURCE_FLAGS_NONE,
    FFX_RESOURCE_STATE_COMPUTE_READ, FFX_RESOURCE_STATE_UNORDERED_ACCESS,
    FFX_RESOURCE_TYPE_TEXTURE2D, FFX_SURFACE_FORMAT_R8G8B8A8_UNORM,
    FFX_SURFACE_FORMAT_R16G16_FLOAT, FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT,
    FFX_SURFACE_FORMAT_R32_FLOAT, FfxCommandList, FfxDevice, FfxDimensions2D, FfxFloatCoords2D,
    FfxFrameInterpolationContext, FfxFrameInterpolationContextDescription,
    FfxFrameInterpolationDispatchDescription, FfxFrameInterpolationPrepareDescription,
    FfxFrameInterpolationSharedResourceDescriptions, FfxInterface, FfxRect2D, FfxResource,
    FfxResourceDescription, FfxResourceState, FfxSurfaceFormat,
};
use super::ffx_fg::*;
use super::vulkan_context::VulkanContext;
// ---------------------------------------------------------------------------
// Helpers — Vulkan image creation
// ---------------------------------------------------------------------------
/// A Vulkan image + memory + view triplet.
struct GpuImage {
    image: vk::Image,
    #[allow(dead_code)]
    memory: vk::DeviceMemory,
    _view: vk::ImageView,
    #[allow(dead_code)]
    format: vk::Format,
    width: u32,
    height: u32,
}
/// Create a 2D image suitable for compute shader read/write.
fn create_compute_image(
    device: &ash::Device,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    format: vk::Format,
    width: u32,
    height: u32,
) -> Result<GpuImage, InterpolateError> {
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
            vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = unsafe { device.create_image(&image_info, None) }
        .map_err(|e| InterpolateError::InterpolateFailed(e.to_string()))?;
    let requirements = unsafe { device.get_image_memory_requirements(image) };
    let type_idx = find_memory_type(
        mem_props,
        requirements,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .ok_or_else(|| InterpolateError::InterpolateFailed("no DEVICE_LOCAL memory type".into()))?;
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(type_idx);
    let memory = unsafe { device.allocate_memory(&alloc_info, None) }
        .map_err(|e| InterpolateError::InterpolateFailed(e.to_string()))?;
    unsafe { device.bind_image_memory(image, memory, 0) }
        .map_err(|e| InterpolateError::InterpolateFailed(e.to_string()))?;
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
    let view = unsafe { device.create_image_view(&view_info, None) }
        .map_err(|e| InterpolateError::InterpolateFailed(e.to_string()))?;
    Ok(GpuImage {
        image,
        memory,
        _view: view,
        format,
        width,
        height,
    })
}
fn find_memory_type(
    mem_props: vk::PhysicalDeviceMemoryProperties,
    requirements: vk::MemoryRequirements,
    required: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..mem_props.memory_type_count).find(|&i| {
        requirements.memory_type_bits & (1 << i) != 0
            && mem_props.memory_types[i as usize]
                .property_flags
                .contains(required)
    })
}
// ---------------------------------------------------------------------------
// VkDeviceContext wrapper for ffxGetDeviceVK
// ---------------------------------------------------------------------------
#[repr(C)]
struct VkDeviceContext {
    vk_device: ash::Device,
    vk_physical_device: vk::PhysicalDevice,
    vk_device_proc_addr: *const std::ffi::c_void,
}
// ---------------------------------------------------------------------------
// FSR3FrameGen — main backend
// ---------------------------------------------------------------------------
/// AMD FSR3 Frame Generation backend.
///
/// Generates an interpolated frame between two consecutive real frames
/// using optical flow + frame interpolation compute shaders from the
/// `FidelityFX` SDK.
pub struct Fsr3FrameGen {
    /// Loaded `FidelityFX` library + function pointers.
    /// Shared Vulkan context (device, queue, command pool).
    vk: Arc<Mutex<VulkanContext>>,
    /// Frame Generation context handle (opaque).
    fg_ctx: FfxFrameInterpolationContext,
    // --- Persistent GPU resources ---
    /// Input color texture (render resolution, RGBA16F).
    input_color: GpuImage,
    /// Input depth texture (render resolution, R32F).
    input_depth: GpuImage,
    /// Input motion vectors (render resolution, RG16F).
    input_mv: GpuImage,
    /// Output interpolated texture (display resolution, RGBA16F).
    output_color: GpuImage,
    /// Dilated depth (shared resource, render resolution, R32F).
    dilated_depth: GpuImage,
    /// Dilated motion vectors (shared resource, render resolution, RG16F).
    dilated_mv: GpuImage,
    /// Reconstructed previous nearest depth (shared resource, R32F).
    reconstructed_prev_depth: vk::Buffer,
    #[allow(dead_code)]
    reconstructed_prev_depth_mem: vk::DeviceMemory,
    /// Optical flow vector texture.
    optical_flow_vector: GpuImage,
    /// Optical flow SCD texture.
    optical_flow_scd: GpuImage,
    /// Staging buffer for GPU→CPU readback.
    staging_buffer: vk::Buffer,
    staging_mem: vk::DeviceMemory,
    staging_size: u64,
    /// Pre-built CPU scratch for the constant depth-far fill (1.0f per pixel,
    /// `render_w * render_h * 4` bytes). Re-uploaded each frame instead of
    /// being recomputed.
    depth_scratch: Vec<u8>,
    /// Pre-built CPU scratch for the constant zero motion-vector fill
    /// (`render_w * render_h * 4` bytes).
    mv_scratch: Vec<u8>,
    /// Frame ID counter.
    frame_id: Mutex<u64>,
    /// Render resolution.
    render_w: u32,
    render_h: u32,
    /// Display (output) resolution.
    display_w: u32,
    display_h: u32,
}
// SAFETY: All Vulkan resources are behind Arc<Mutex<>>.
// SAFETY: Vulkan state is accessed only via the serialized FFX context.
unsafe impl Send for Fsr3FrameGen {}
// SAFETY: Vulkan state is accessed only via the serialized FFX context.
unsafe impl Sync for Fsr3FrameGen {}
impl Fsr3FrameGen {
    /// Create a new FSR3 Frame Generation backend.
    ///
    /// Loads the `FidelityFX` SDK, initializes Vulkan backend, creates the FG
    /// context and persistent GPU resources.
    ///
    /// # Arguments
    /// * `render_w`, `render_h` — render resolution (input frames).
    /// * `display_w`, `display_h` — display/output resolution.
    pub fn new(
        render_w: u32,
        render_h: u32,
        display_w: u32,
        display_h: u32,
    ) -> Result<Self, InterpolateError> {
        
        // 2. Create Vulkan context.
        let vk = Arc::new(Mutex::new(VulkanContext::new(&[])?));
        let guard = vk
            .lock()
            .map_err(|e| InterpolateError::InitFailed(format!("mutex: {e}")))?;
        let device = guard.device.clone();
        let physical_device = guard.physical_device;
        // 3. Get Vulkan backend interface.
        let max_contexts = 1u32;
        let scratch_size =
            unsafe { ffxGetScratchMemorySizeVK(physical_device, max_contexts) };
        let mut scratch = vec![0u8; scratch_size];
        let mut backend_interface = FfxInterface::default();
        let vk_ctx = VkDeviceContext {
            vk_device: device.clone(),
            vk_physical_device: physical_device,
            vk_device_proc_addr: std::ptr::null(),
        };
        let ffx_device: FfxDevice = &vk_ctx as *const VkDeviceContext as *mut std::ffi::c_void;
        let err = unsafe {
            ffxGetInterfaceVK(
                &mut backend_interface,
                ffx_device,
                scratch.as_mut_ptr().cast(),
                scratch_size,
                max_contexts,
            )
        };
        if err != FFX_OK {
            return Err(InterpolateError::InitFailed(format!(
                "ffxGetInterfaceVK failed: {err}"
            )));
        }
        // 4. Create Frame Generation context.
        //    Note: `backend_interface` is moved into the context description here.
        //    After `ffxFrameInterpolationContextCreate` returns successfully the
        //    interface contents are copied internally by the SDK; we do not need
        //    to keep our own copy.
        let context_desc = FfxFrameInterpolationContextDescription {
            flags: FFX_FRAMEINTERPOLATION_ENABLE_DEPTH_INVERTED
                | FFX_FRAMEINTERPOLATION_ENABLE_DEPTH_INFINITE
                | FFX_FRAMEINTERPOLATION_ENABLE_HDR_COLOR_INPUT
                | FFX_FRAMEINTERPOLATION_ENABLE_JITTER_MOTION_VECTORS,
            max_render_size: FfxDimensions2D {
                width: render_w,
                height: render_h,
            },
            display_size: FfxDimensions2D {
                width: display_w,
                height: display_h,
            },
            backend_interface,
            back_buffer_format: FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT,
            previous_interpolation_source_format: FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT,
        };
        let mut fg_ctx: FfxFrameInterpolationContext = std::ptr::null_mut();
        let err = unsafe { ffxFrameInterpolationContextCreate(&mut fg_ctx, &context_desc) };
        if err != FFX_OK {
            return Err(InterpolateError::InitFailed(format!(
                "ffxFrameInterpolationContextCreate failed: {err}"
            )));
        }
        // 5. Get shared resource descriptions and create resources.
        let mut shared_desc = FfxFrameInterpolationSharedResourceDescriptions {
            dilated_depth: FfxResourceDescription::default(),
            dilated_motion_vectors: FfxResourceDescription::default(),
            reconstructed_prev_nearest_depth: FfxResourceDescription::default(),
        };
        let err = unsafe { ffxFrameInterpolationGetSharedResourceDescriptions(&fg_ctx, &mut shared_desc) };
        if err != FFX_OK {
            unsafe { ffxFrameInterpolationContextDestroy(&mut fg_ctx) };
            return Err(InterpolateError::InitFailed(format!(
                "ffxFrameInterpolationGetSharedResourceDescriptions failed: {err}"
            )));
        }
        let mem_props = unsafe {
            guard
                .instance
                .get_physical_device_memory_properties(physical_device)
        };
        // Dilated depth: R32_FLOAT at render resolution.
        let dilated_depth = create_compute_image(
            &device,
            mem_props,
            vk::Format::R32_SFLOAT,
            shared_desc.dilated_depth.width,
            shared_desc.dilated_depth.height,
        )
        .map_err(|e| InterpolateError::InitFailed(format!("dilated_depth image: {e}")))?;
        // Dilated motion vectors: R16G16_FLOAT at render resolution.
        let dilated_mv = create_compute_image(
            &device,
            mem_props,
            vk::Format::R16G16_SFLOAT,
            shared_desc.dilated_motion_vectors.width,
            shared_desc.dilated_motion_vectors.height,
        )
        .map_err(|e| InterpolateError::InitFailed(format!("dilated_mv image: {e}")))?;
        // Reconstructed prev depth: buffer (render_w * render_h * 4 bytes).
        let rec_depth_size = shared_desc.reconstructed_prev_nearest_depth.width as u64
            * shared_desc.reconstructed_prev_nearest_depth.height as u64
            * 4;
        let (rec_buf, rec_mem) = guard
            .create_buffer(
                rec_depth_size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .map_err(|e| {
                InterpolateError::InitFailed(format!("reconstructed_prev_depth buffer: {e}"))
            })?;
        // Input color: RGBA16F at render resolution.
        let input_color = create_compute_image(
            &device,
            mem_props,
            vk::Format::R16G16B16A16_SFLOAT,
            render_w,
            render_h,
        )
        .map_err(|e| InterpolateError::InitFailed(format!("input_color: {e}")))?;
        // Input depth: R32F at render resolution.
        let input_depth = create_compute_image(
            &device,
            mem_props,
            vk::Format::R32_SFLOAT,
            render_w,
            render_h,
        )
        .map_err(|e| InterpolateError::InitFailed(format!("input_depth: {e}")))?;
        // Input motion vectors: RG16F at render resolution.
        let input_mv = create_compute_image(
            &device,
            mem_props,
            vk::Format::R16G16_SFLOAT,
            render_w,
            render_h,
        )
        .map_err(|e| InterpolateError::InitFailed(format!("input_mv: {e}")))?;
        // Output color: RGBA16F at display resolution.
        let output_color = create_compute_image(
            &device,
            mem_props,
            vk::Format::R16G16B16A16_SFLOAT,
            display_w,
            display_h,
        )
        .map_err(|e| InterpolateError::InitFailed(format!("output_color: {e}")))?;
        // Optical flow vector: RG16F at render resolution.
        let optical_flow_vector = create_compute_image(
            &device,
            mem_props,
            vk::Format::R16G16_SFLOAT,
            render_w,
            render_h,
        )
        .map_err(|e| InterpolateError::InitFailed(format!("optical_flow_vector: {e}")))?;
        // Optical flow SCD: R8_UNORM at render resolution.
        let optical_flow_scd =
            create_compute_image(&device, mem_props, vk::Format::R8_UNORM, render_w, render_h)
                .map_err(|e| InterpolateError::InitFailed(format!("optical_flow_scd: {e}")))?;
        // Staging buffer for readback: display resolution × 8 bytes (RGBA16F).
        let staging_size = display_w as u64 * display_h as u64 * 8;
        let (staging_buffer, staging_mem) = guard
            .create_buffer(
                staging_size,
                vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT
                    | vk::MemoryPropertyFlags::HOST_CACHED,
            )
            .map_err(|e| InterpolateError::InitFailed(format!("staging buffer: {e}")))?;
        drop(guard);

        // Build constant CPU scratch buffers once. `depth` is a fill of
        // `1.0f32` (far plane under inverted depth) and `mv` is zeros — they
        // never change between frames, only the GPU image gets re-uploaded.
        let pixels = render_w as usize * render_h as usize;
        let depth_bytes = f32::to_bits(1.0f32).to_le_bytes();
        let mut depth_scratch = Vec::with_capacity(pixels * 4);
        for _ in 0..pixels {
            depth_scratch.extend_from_slice(&depth_bytes);
        }
        let mv_scratch = vec![0u8; pixels * 4];

        Ok(Self {
            vk,
            fg_ctx,
            input_color,
            input_depth,
            input_mv,
            output_color,
            dilated_depth,
            dilated_mv,
            reconstructed_prev_depth: rec_buf,
            reconstructed_prev_depth_mem: rec_mem,
            optical_flow_vector,
            optical_flow_scd,
            staging_buffer,
            staging_mem,
            staging_size,
            depth_scratch,
            mv_scratch,
            frame_id: Mutex::new(0),
            render_w,
            render_h,
            display_w,
            display_h,
        })
    }
    /// Generate an interpolated frame between `a` (previous) and `b` (current).
    ///
    /// The interpolation happens at the temporal midpoint (t=0.5).
    /// Frame `b` is the "current back buffer" for the FG dispatch.
    /// Frame `a` provides the optical flow source.
    pub fn generate_frame(&mut self, a: &GpuFrame, b: &GpuFrame) -> Result<GpuFrame, InterpolateError> {
        if !a.same_dimensions(b) {
            return Err(InterpolateError::DimensionMismatch(
                a.width, a.height, b.width, b.height,
            ));
        }
        let guard = self
            .vk
            .lock()
            .map_err(|e| InterpolateError::InterpolateFailed(format!("mutex: {e}")))?;
        let device = &guard.device;
        // Upload frame `b` as current color (RGBA8 → RGBA16F).
        upload_rgba8_to_rgba16f(
            device,
            &guard,
            &b.data,
            b.width,
            b.height,
            b.stride,
            &self.input_color,
        )?;
        // Fill depth with far value (1.0 for inverted depth with infinite far plane).
        fill_depth_far(device, &guard, &self.input_depth, &self.depth_scratch)?;
        // Fill motion vectors with zero (no engine-provided MV for captured content).
        fill_mv_zero(device, &guard, &self.input_mv, &self.mv_scratch)?;
        // Allocate command buffer and begin recording.
        let cmd = guard
            .allocate_command_buffer()
            .map_err(|e| InterpolateError::InterpolateFailed(format!("alloc cmd: {e}")))?;
        let begin_info = vk::CommandBufferBeginInfo::default();
        unsafe { device.begin_command_buffer(cmd, &begin_info) }
            .map_err(|e| InterpolateError::InterpolateFailed(format!("begin cmd: {e}")))?;
        // Transition input images to COMPUTE_READ.
        cmd_image_barrier(
            device,
            cmd,
            self.input_color.image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
        );
        cmd_image_barrier(
            device,
            cmd,
            self.input_depth.image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
        );
        cmd_image_barrier(
            device,
            cmd,
            self.input_mv.image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
        );
        // Transition shared resources to COMPUTE_READ.
        cmd_image_barrier(
            device,
            cmd,
            self.dilated_depth.image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
        );
        cmd_image_barrier(
            device,
            cmd,
            self.dilated_mv.image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
        );
        cmd_image_barrier(
            device,
            cmd,
            self.optical_flow_vector.image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
        );
        cmd_image_barrier(
            device,
            cmd,
            self.optical_flow_scd.image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
        );
        // Output image: transition to GENERAL for UAV writes.
        cmd_image_barrier(
            device,
            cmd,
            self.output_color.image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::SHADER_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
        );
        // Build FfxResources from Vulkan images.
        let ffx_cmd: FfxCommandList = cmd.as_raw() as *mut std::ffi::c_void;
        let fx_depth = ffx_resource(
            self.input_depth.image,
            FFX_SURFACE_FORMAT_R32_FLOAT,
            self.render_w,
            self.render_h,
            FFX_RESOURCE_STATE_COMPUTE_READ,
        );
        let fx_mv = ffx_resource(
            self.input_mv.image,
            FFX_SURFACE_FORMAT_R16G16_FLOAT,
            self.render_w,
            self.render_h,
            FFX_RESOURCE_STATE_COMPUTE_READ,
        );
        let fx_dilated_depth = ffx_resource(
            self.dilated_depth.image,
            FFX_SURFACE_FORMAT_R32_FLOAT,
            self.render_w,
            self.render_h,
            FFX_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        let fx_dilated_mv = ffx_resource(
            self.dilated_mv.image,
            FFX_SURFACE_FORMAT_R16G16_FLOAT,
            self.render_w,
            self.render_h,
            FFX_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        let fx_rec_prev_depth = FfxResource {
            resource: self.reconstructed_prev_depth.as_raw() as *mut std::ffi::c_void,
            description: FfxResourceDescription {
                type_: 0, // BUFFER
                format: 0,
                width: self.render_w * self.render_h,
                height: 1,
                depth: 1,
                mip_count: 1,
                flags: FFX_RESOURCE_FLAGS_NONE,
                usage: 2, // UAV
            },
            state: FFX_RESOURCE_STATE_UNORDERED_ACCESS,
        };
        // --- Prepare pass ---
        let frame_id = {
            let mut fid = self.frame_id.lock().unwrap();
            let id = *fid;
            *fid = id.wrapping_add(1);
            id
        };
        let prepare_desc = FfxFrameInterpolationPrepareDescription {
            flags: 0,
            command_list: ffx_cmd,
            render_size: FfxDimensions2D {
                width: self.render_w,
                height: self.render_h,
            },
            jitter_offset: FfxFloatCoords2D { x: 0.0, y: 0.0 },
            motion_vector_scale: FfxFloatCoords2D { x: 1.0, y: 1.0 },
            frame_time_delta: 16.67, // 60fps
            camera_near: 0.01,
            camera_far: 1000.0,
            camera_fov_angle_vertical: 1.0,
            view_space_to_meters_factor: 1.0,
            depth: fx_depth,
            motion_vectors: fx_mv,
            frame_id,
            dilated_depth: fx_dilated_depth,
            dilated_motion_vectors: fx_dilated_mv,
            reconstructed_prev_depth: fx_rec_prev_depth,
        };
        let err = unsafe { ffxFrameInterpolationPrepare(&mut self.fg_ctx, &prepare_desc) };
        if err != FFX_OK {
            unsafe {
                let _ = device.end_command_buffer(cmd);
                device.free_command_buffers(guard.command_pool, &[cmd]);
            }
            drop(guard);
            return Err(InterpolateError::InterpolateFailed(format!(
                "ffxFrameInterpolationPrepare failed: {err}"
            )));
        }
        // --- Dispatch pass ---
        let fx_input = ffx_resource(
            self.input_color.image,
            FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT,
            self.render_w,
            self.render_h,
            FFX_RESOURCE_STATE_COMPUTE_READ,
        );
        let fx_output = ffx_resource(
            self.output_color.image,
            FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT,
            self.display_w,
            self.display_h,
            FFX_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        let fx_of_vector = ffx_resource(
            self.optical_flow_vector.image,
            FFX_SURFACE_FORMAT_R16G16_FLOAT,
            self.render_w,
            self.render_h,
            FFX_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        let fx_of_scd = ffx_resource(
            self.optical_flow_scd.image,
            FFX_SURFACE_FORMAT_R8G8B8A8_UNORM,
            self.render_w,
            self.render_h,
            FFX_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        let dispatch_desc = FfxFrameInterpolationDispatchDescription {
            flags: FFX_FRAMEINTERPOLATION_DISPATCH_DRAW_DEBUG_TEAR_LINES,
            command_list: ffx_cmd,
            display_size: FfxDimensions2D {
                width: self.display_w,
                height: self.display_h,
            },
            render_size: FfxDimensions2D {
                width: self.render_w,
                height: self.render_h,
            },
            current_back_buffer: fx_input,
            current_back_buffer_hudless: FfxResource::default(),
            output: fx_output,
            interpolation_rect: FfxRect2D {
                left: 0,
                top: 0,
                width: self.display_w as i32,
                height: self.display_h as i32,
            },
            optical_flow_vector: fx_of_vector,
            optical_flow_scene_change_detection: fx_of_scd,
            optical_flow_scale: FfxFloatCoords2D { x: 1.0, y: 1.0 },
            optical_flow_block_size: 8,
            camera_near: 0.01,
            camera_far: 1000.0,
            camera_fov_angle_vertical: 1.0,
            view_space_to_meters_factor: 1.0,
            frame_time_delta: 16.67,
            reset: false,
            backbuffer_transfer_function: FFX_BACKBUFFER_TRANSFER_FUNCTION_SRGB,
            min_max_luminance: FfxFloatCoords2D {
                x: 0.0001,
                y: 1000.0,
            },
            frame_id,
            dilated_depth: fx_dilated_depth,
            dilated_motion_vectors: fx_dilated_mv,
            reconstructed_prev_depth: fx_rec_prev_depth,
            distortion_field: FfxResource::default(),
        };
        let err = unsafe { ffxFrameInterpolationDispatch(&mut self.fg_ctx, &dispatch_desc) };
        if err != FFX_OK {
            unsafe {
                let _ = device.end_command_buffer(cmd);
                device.free_command_buffers(guard.command_pool, &[cmd]);
            }
            drop(guard);
            return Err(InterpolateError::InterpolateFailed(format!(
                "ffxFrameInterpolationDispatch failed: {err}"
            )));
        }
        // Transition output for copy.
        cmd_image_barrier(
            device,
            cmd,
            self.output_color.image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
        );
        // Copy output → staging buffer.
        unsafe {
            device.cmd_copy_image_to_buffer(
                cmd,
                self.output_color.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.staging_buffer,
                &[vk::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                    image_extent: vk::Extent3D {
                        width: self.display_w,
                        height: self.display_h,
                        depth: 1,
                    },
                }],
            );
        }
        unsafe { device.end_command_buffer(cmd) }
            .map_err(|e| InterpolateError::InterpolateFailed(format!("end cmd: {e}")))?;
        // Submit and wait.
        guard
            .submit_and_wait(cmd)
            .map_err(|e| InterpolateError::InterpolateFailed(format!("submit: {e}")))?;
        // Free command buffer.
        unsafe {
            device.free_command_buffers(guard.command_pool, &[cmd]);
        }
        // Read back from staging buffer.
        let data = read_rgba16f_to_rgba8(
            device,
            &self.staging_mem,
            self.staging_size as usize,
            self.display_w,
            self.display_h,
        )?;
        drop(guard);
        // Timestamp: midpoint.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let ts = if b.timestamp_ns >= a.timestamp_ns {
            a.timestamp_ns + (b.timestamp_ns - a.timestamp_ns) / 2
        } else {
            a.timestamp_ns
        };
        Ok(GpuFrame {
            data,
            width: self.display_w,
            height: self.display_h,
            stride: self.display_w * 4,
            timestamp_ns: ts,
        })
    }
}
impl FrameInterpolator for Fsr3FrameGen {
    fn interpolate(
        &mut self,
        a: &GpuFrame,
        b: &GpuFrame,
        _t: f32,
    ) -> Result<GpuFrame, InterpolateError> {
        self.generate_frame(a, b)
    }
    fn latency_ms(&self) -> f32 {
        16.67 // ~1 frame at 60fps
    }
    fn name(&self) -> &'static str {
        "fsr3-fg"
    }
}
impl Drop for Fsr3FrameGen {
    fn drop(&mut self) {
        if !self.fg_ctx.is_null() {
            unsafe { ffxFrameInterpolationContextDestroy(&mut self.fg_ctx) };
        }
        // Vulkan resources are cleaned up via VulkanContext::drop.
    }
}
// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
fn ffx_resource(
    image: vk::Image,
    format: FfxSurfaceFormat,
    width: u32,
    height: u32,
    state: FfxResourceState,
) -> FfxResource {
    FfxResource {
        resource: image.as_raw() as *mut std::ffi::c_void,
        description: FfxResourceDescription {
            type_: FFX_RESOURCE_TYPE_TEXTURE2D,
            format,
            width,
            height,
            depth: 1,
            mip_count: 1,
            flags: FFX_RESOURCE_FLAGS_NONE,
            usage: 2, // UAV
        },
        state,
    }
}
#[allow(clippy::too_many_arguments)]
fn cmd_image_barrier(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_access: vk::AccessFlags,
    dst_access: vk::AccessFlags,
    src_stage: vk::PipelineStageFlags,
    dst_stage: vk::PipelineStageFlags,
) {
    let barrier = vk::ImageMemoryBarrier::default()
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)
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
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }
}
/// Upload RGBA8 frame data to an RGBA16F Vulkan image via staging buffer.
fn upload_rgba8_to_rgba16f(
    device: &ash::Device,
    vk_ctx: &VulkanContext,
    data: &[u8],
    width: u32,
    height: u32,
    stride: u32,
    dst_image: &GpuImage,
) -> Result<(), InterpolateError> {
    // Convert RGBA8 → RGBA16F (half-float, 8 bytes/pixel).
    let row_bytes = width as usize * 8;
    let total = row_bytes * height as usize;
    let mut f16: Vec<u8> = vec![0u8; total];
    for y in 0..height as usize {
        let src_row = y * stride as usize;
        let dst_row = y * row_bytes;
        for x in 0..width as usize {
            let src = src_row + x * 4;
            let dst = dst_row + x * 8;
            // Convert each 8-bit channel to f16 (using simple /255.0 conversion).
            for c in 0..3 {
                let val = f32::from(data[src + c]) / 255.0;
                let half = half::f16::from_f32(val);
                let bytes = half.to_le_bytes();
                f16[dst + c * 2] = bytes[0];
                f16[dst + c * 2 + 1] = bytes[1];
            }
            // Alpha: same conversion.
            let alpha = f32::from(data[src + 3]) / 255.0;
            let half_a = half::f16::from_f32(alpha);
            let a_bytes = half_a.to_le_bytes();
            f16[dst + 6] = a_bytes[0];
            f16[dst + 7] = a_bytes[1];
        }
    }
    let (staging_buf, staging_mem) = vk_ctx.create_buffer(
        total as u64,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    vk_ctx.upload_to_buffer(staging_mem, &f16)?;
    let cmd = vk_ctx.allocate_command_buffer()?;
    let begin_info = vk::CommandBufferBeginInfo::default();
    unsafe { device.begin_command_buffer(cmd, &begin_info) }
        .map_err(|e| InterpolateError::InterpolateFailed(e.to_string()))?;
    // Transition dst to TRANSFER_DST.
    cmd_image_barrier(
        device,
        cmd,
        dst_image.image,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::AccessFlags::empty(),
        vk::AccessFlags::TRANSFER_WRITE,
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::PipelineStageFlags::TRANSFER,
    );
    unsafe {
        device.cmd_copy_buffer_to_image(
            cmd,
            staging_buf,
            dst_image.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                image_extent: vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                },
            }],
        );
    }
    unsafe { device.end_command_buffer(cmd) }
        .map_err(|e| InterpolateError::InterpolateFailed(e.to_string()))?;
    vk_ctx.submit_and_wait(cmd)?;
    unsafe {
        device.destroy_buffer(staging_buf, None);
        device.free_memory(staging_mem, None);
        device.free_command_buffers(vk_ctx.command_pool, &[cmd]);
    }
    Ok(())
}
/// Fill a depth image with the far-plane value (1.0 for inverted+depth infinite).
fn fill_depth_far(
    device: &ash::Device,
    vk_ctx: &VulkanContext,
    dst_image: &GpuImage,
    data: &[u8],
) -> Result<(), InterpolateError> {
    let size = dst_image.width as u64 * dst_image.height as u64 * 4;
    debug_assert_eq!(data.len(), size as usize, "depth scratch size mismatch");
    let (staging, staging_mem) = vk_ctx.create_buffer(
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    vk_ctx.upload_to_buffer(staging_mem, data)?;
    let cmd = vk_ctx.allocate_command_buffer()?;
    unsafe { device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()) }
        .map_err(|e| InterpolateError::InterpolateFailed(e.to_string()))?;
    cmd_image_barrier(
        device,
        cmd,
        dst_image.image,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::AccessFlags::empty(),
        vk::AccessFlags::TRANSFER_WRITE,
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::PipelineStageFlags::TRANSFER,
    );
    unsafe {
        device.cmd_copy_buffer_to_image(
            cmd,
            staging,
            dst_image.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                image_extent: vk::Extent3D {
                    width: dst_image.width,
                    height: dst_image.height,
                    depth: 1,
                },
            }],
        );
    }
    unsafe { device.end_command_buffer(cmd) }
        .map_err(|e| InterpolateError::InterpolateFailed(e.to_string()))?;
    vk_ctx.submit_and_wait(cmd)?;
    unsafe {
        device.destroy_buffer(staging, None);
        device.free_memory(staging_mem, None);
        device.free_command_buffers(vk_ctx.command_pool, &[cmd]);
    }
    Ok(())
}
/// Fill a motion vector image with zero.
fn fill_mv_zero(
    device: &ash::Device,
    vk_ctx: &VulkanContext,
    dst_image: &GpuImage,
    data: &[u8],
) -> Result<(), InterpolateError> {
    let size = dst_image.width as u64 * dst_image.height as u64 * 4; // RG16F = 4 bytes
    debug_assert_eq!(data.len(), size as usize, "mv scratch size mismatch");
    let (staging, staging_mem) = vk_ctx.create_buffer(
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    vk_ctx.upload_to_buffer(staging_mem, data)?;
    let cmd = vk_ctx.allocate_command_buffer()?;
    unsafe { device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()) }
        .map_err(|e| InterpolateError::InterpolateFailed(e.to_string()))?;
    cmd_image_barrier(
        device,
        cmd,
        dst_image.image,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::AccessFlags::empty(),
        vk::AccessFlags::TRANSFER_WRITE,
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::PipelineStageFlags::TRANSFER,
    );
    unsafe {
        device.cmd_copy_buffer_to_image(
            cmd,
            staging,
            dst_image.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                image_extent: vk::Extent3D {
                    width: dst_image.width,
                    height: dst_image.height,
                    depth: 1,
                },
            }],
        );
    }
    unsafe { device.end_command_buffer(cmd) }
        .map_err(|e| InterpolateError::InterpolateFailed(e.to_string()))?;
    vk_ctx.submit_and_wait(cmd)?;
    unsafe {
        device.destroy_buffer(staging, None);
        device.free_memory(staging_mem, None);
        device.free_command_buffers(vk_ctx.command_pool, &[cmd]);
    }
    Ok(())
}
/// Read RGBA16F image data from staging buffer, converting back to RGBA8.
fn read_rgba16f_to_rgba8(
    device: &ash::Device,
    staging_mem: &vk::DeviceMemory,
    staging_size: usize,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, InterpolateError> {
    let ptr = unsafe {
        device
            .map_memory(
                *staging_mem,
                0,
                staging_size as u64,
                vk::MemoryMapFlags::empty(),
            )
            .map_err(|e| InterpolateError::InterpolateFailed(format!("map memory: {e}")))?
    } as *const u8;
    let f16_slice = unsafe { std::slice::from_raw_parts(ptr, staging_size) };
    let row_bytes = width as usize * 8;
    let out_row = width as usize * 4;
    let mut out = vec![0u8; out_row * height as usize];
    for y in 0..height as usize {
        let src_row = y * row_bytes;
        let dst_row = y * out_row;
        for x in 0..width as usize {
            let src = src_row + x * 8;
            let dst = dst_row + x * 4;
            for c in 0..4 {
                let half_bytes = [f16_slice[src + c * 2], f16_slice[src + c * 2 + 1]];
                let half = half::f16::from_le_bytes(half_bytes);
                let val = half.to_f32();
                out[dst + c] = (val * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    unsafe { device.unmap_memory(*staging_mem) };
    Ok(out)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fsr3_fg_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Fsr3FrameGen>();
    }
}

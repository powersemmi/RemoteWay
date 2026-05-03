//! FFI bindings for AMD `FidelityFX` Frame Generation (`ffxFrameInterpolation*`).
//!
//! The SDK C++ source is compiled into a static library at build time via
//! `build.rs` + the `cc` crate, then linked directly. No dynamic loading needed.
//!
//! ## SDK Source
//!
//! The FidelityFX SDK v1.1.4 is cloned from GitHub at build time and the
//! frameinterpolation component is compiled with GCC/Clang (same approach
//! as the `fsr-sys` crate).

#![allow(clippy::undocumented_unsafe_blocks)]

use std::os::raw::c_void;

use ash::vk;
use ash::vk::Handle;

// ---------------------------------------------------------------------------
// Opaque context handle
// ---------------------------------------------------------------------------

/// Opaque FSR3 Frame Interpolation context (heap-allocated by the SDK).
pub type FfxFrameInterpolationContext = *mut c_void;

/// `FidelityFX` error code (0 = success).
pub type FfxErrorCode = i32;

/// Success code.
pub const FFX_OK: FfxErrorCode = 0;

// ---------------------------------------------------------------------------
// Shared Vulkan types (compatible with fidelityfx-rs layout)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct FfxDimensions2D {
    pub width: u32,
    pub height: u32,
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct FfxFloatCoords2D {
    pub x: f32,
    pub y: f32,
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct FfxRect2D {
    pub left: i32,
    pub top: i32,
    pub width: i32,
    pub height: i32,
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct FfxEffectMemoryUsage {
    pub total_usage_in_bytes: u64,
    pub aliasable_usage_in_bytes: u64,
}

// ---------------------------------------------------------------------------
// Resource types
// ---------------------------------------------------------------------------

pub type FfxResourceType = u32;
pub const FFX_RESOURCE_TYPE_BUFFER: FfxResourceType = 0;
pub const FFX_RESOURCE_TYPE_TEXTURE1D: FfxResourceType = 1;
pub const FFX_RESOURCE_TYPE_TEXTURE2D: FfxResourceType = 2;

pub type FfxSurfaceFormat = u32;
pub const FFX_SURFACE_FORMAT_R8G8B8A8_UNORM: FfxSurfaceFormat = 10;
pub const FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT: FfxSurfaceFormat = 4;
pub const FFX_SURFACE_FORMAT_R32G32B32A32_FLOAT: FfxSurfaceFormat = 3;
pub const FFX_SURFACE_FORMAT_R32_FLOAT: FfxSurfaceFormat = 28;
pub const FFX_SURFACE_FORMAT_R16G16_FLOAT: FfxSurfaceFormat = 18;
pub const FFX_SURFACE_FORMAT_R8_UNORM: FfxSurfaceFormat = 25;

pub type FfxResourceFlags = u32;
pub const FFX_RESOURCE_FLAGS_NONE: FfxResourceFlags = 0;
pub const FFX_RESOURCE_FLAGS_ALIASABLE: FfxResourceFlags = 1;

pub type FfxResourceUsage = u32;
pub const FFX_RESOURCE_USAGE_READ_ONLY: FfxResourceUsage = 0;
pub const FFX_RESOURCE_USAGE_RENDERTARGET: FfxResourceUsage = 1;
pub const FFX_RESOURCE_USAGE_UAV: FfxResourceUsage = 2;

pub type FfxResourceState = u32;
pub const FFX_RESOURCE_STATE_UNORDERED_ACCESS: FfxResourceState = 2;
pub const FFX_RESOURCE_STATE_COMPUTE_READ: FfxResourceState = 4;
pub const FFX_RESOURCE_STATE_PIXEL_COMPUTE_READ: FfxResourceState = 12;
pub const FFX_RESOURCE_STATE_COPY_SRC: FfxResourceState = 16;
pub const FFX_RESOURCE_STATE_COPY_DEST: FfxResourceState = 32;
pub const FFX_RESOURCE_STATE_GENERIC_READ: FfxResourceState = 20;

pub type FfxBackbufferTransferFunction = u32;
pub const FFX_BACKBUFFER_TRANSFER_FUNCTION_SRGB: FfxBackbufferTransferFunction = 0;
pub const FFX_BACKBUFFER_TRANSFER_FUNCTION_PQ: FfxBackbufferTransferFunction = 1;

// ---------------------------------------------------------------------------
// Resource description (layout-compatible with fidelityfx-rs)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct FfxResourceDescription {
    pub type_: FfxResourceType,
    pub format: FfxSurfaceFormat,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub mip_count: u32,
    pub flags: FfxResourceFlags,
    pub usage: FfxResourceUsage,
}

impl Default for FfxResourceDescription {
    fn default() -> Self {
        Self {
            type_: FFX_RESOURCE_TYPE_TEXTURE2D,
            format: 0,
            width: 0,
            height: 0,
            depth: 1,
            mip_count: 1,
            flags: FFX_RESOURCE_FLAGS_NONE,
            usage: FFX_RESOURCE_USAGE_READ_ONLY,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct FfxResource {
    pub resource: *mut c_void,
    pub description: FfxResourceDescription,
    pub state: FfxResourceState,
}

impl Default for FfxResource {
    fn default() -> Self {
        Self {
            resource: std::ptr::null_mut(),
            description: FfxResourceDescription::default(),
            state: FFX_RESOURCE_STATE_COMPUTE_READ,
        }
    }
}

/// Wraps a Vulkan image as an [`FfxResource`].
///
/// # Safety
///
/// The image must be valid and not destroyed while the resource is in use.
#[must_use]
pub unsafe fn make_resource_vk(
    image: vk::Image,
    format: FfxSurfaceFormat,
    width: u32,
    height: u32,
    state: FfxResourceState,
) -> FfxResource {
    FfxResource {
        resource: image.as_raw() as *mut c_void,
        description: FfxResourceDescription {
            type_: FFX_RESOURCE_TYPE_TEXTURE2D,
            format,
            width,
            height,
            depth: 1,
            mip_count: 1,
            flags: FFX_RESOURCE_FLAGS_NONE,
            usage: FFX_RESOURCE_USAGE_UAV | FFX_RESOURCE_USAGE_RENDERTARGET,
        },
        state,
    }
}

// ---------------------------------------------------------------------------
// FfxInterface — Vulkan backend function table (populated by ffxGetInterfaceVK)
// ---------------------------------------------------------------------------

/// Opaque `FidelityFX` device handle.
pub type FfxDevice = *mut c_void;

/// Opaque `FidelityFX` command list (`VkCommandBuffer` for Vulkan).
pub type FfxCommandList = *mut c_void;

// FfxInterface is an opaque struct filled in by ffxGetInterfaceVK.
// We declare it with enough storage (size from SDK headers: ~300 pointers).
const FFX_INTERFACE_SIZE: usize = 2048;

#[repr(C, align(8))]
pub struct FfxInterface {
    _opaque: [u8; FFX_INTERFACE_SIZE],
}

impl Default for FfxInterface {
    fn default() -> Self {
        Self {
            _opaque: [0u8; FFX_INTERFACE_SIZE],
        }
    }
}

// ---------------------------------------------------------------------------
// Frame Generation context description
// ---------------------------------------------------------------------------

pub type FfxFrameInterpolationFlags = u32;
pub const FFX_FRAMEINTERPOLATION_ENABLE_DEPTH_INVERTED: FfxFrameInterpolationFlags = 1 << 0;
pub const FFX_FRAMEINTERPOLATION_ENABLE_DEPTH_INFINITE: FfxFrameInterpolationFlags = 1 << 1;
pub const FFX_FRAMEINTERPOLATION_ENABLE_HDR_COLOR_INPUT: FfxFrameInterpolationFlags = 1 << 2;
pub const FFX_FRAMEINTERPOLATION_ENABLE_JITTER_MOTION_VECTORS: FfxFrameInterpolationFlags = 1 << 3;
pub const FFX_FRAMEINTERPOLATION_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS:
    FfxFrameInterpolationFlags = 1 << 4;

#[repr(C)]
pub struct FfxFrameInterpolationContextDescription {
    pub flags: FfxFrameInterpolationFlags,
    pub max_render_size: FfxDimensions2D,
    pub display_size: FfxDimensions2D,
    pub backend_interface: FfxInterface,
    pub back_buffer_format: FfxSurfaceFormat,
    pub previous_interpolation_source_format: FfxSurfaceFormat,
}

// ---------------------------------------------------------------------------
// Shared resource descriptions
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug)]
pub struct FfxFrameInterpolationSharedResourceDescriptions {
    pub dilated_depth: FfxResourceDescription,
    pub dilated_motion_vectors: FfxResourceDescription,
    pub reconstructed_prev_nearest_depth: FfxResourceDescription,
}

// ---------------------------------------------------------------------------
// Prepare description
// ---------------------------------------------------------------------------

pub type FfxFrameInterpolationPrepareFlags = u32;

#[repr(C)]
pub struct FfxFrameInterpolationPrepareDescription {
    pub flags: FfxFrameInterpolationPrepareFlags,
    pub command_list: FfxCommandList,
    pub render_size: FfxDimensions2D,
    pub jitter_offset: FfxFloatCoords2D,
    pub motion_vector_scale: FfxFloatCoords2D,
    pub frame_time_delta: f32,
    pub camera_near: f32,
    pub camera_far: f32,
    pub camera_fov_angle_vertical: f32,
    pub view_space_to_meters_factor: f32,
    pub depth: FfxResource,
    pub motion_vectors: FfxResource,
    pub frame_id: u64,
    pub dilated_depth: FfxResource,
    pub dilated_motion_vectors: FfxResource,
    pub reconstructed_prev_depth: FfxResource,
}

// ---------------------------------------------------------------------------
// Dispatch description
// ---------------------------------------------------------------------------

pub type FfxFrameInterpolationDispatchFlags = u32;
pub const FFX_FRAMEINTERPOLATION_DISPATCH_DRAW_DEBUG_TEAR_LINES:
    FfxFrameInterpolationDispatchFlags = 1 << 0;
pub const FFX_FRAMEINTERPOLATION_DISPATCH_DRAW_DEBUG_VIEW: FfxFrameInterpolationDispatchFlags =
    1 << 2;

#[repr(C)]
pub struct FfxFrameInterpolationDispatchDescription {
    pub flags: FfxFrameInterpolationDispatchFlags,
    pub command_list: FfxCommandList,
    pub display_size: FfxDimensions2D,
    pub render_size: FfxDimensions2D,
    pub current_back_buffer: FfxResource,
    pub current_back_buffer_hudless: FfxResource,
    pub output: FfxResource,
    pub interpolation_rect: FfxRect2D,
    pub optical_flow_vector: FfxResource,
    pub optical_flow_scene_change_detection: FfxResource,
    pub optical_flow_scale: FfxFloatCoords2D,
    pub optical_flow_block_size: u32,
    pub camera_near: f32,
    pub camera_far: f32,
    pub camera_fov_angle_vertical: f32,
    pub view_space_to_meters_factor: f32,
    pub frame_time_delta: f32,
    pub reset: bool,
    pub backbuffer_transfer_function: FfxBackbufferTransferFunction,
    pub min_max_luminance: FfxFloatCoords2D,
    pub frame_id: u64,
    pub dilated_depth: FfxResource,
    pub dilated_motion_vectors: FfxResource,
    pub reconstructed_prev_depth: FfxResource,
    pub distortion_field: FfxResource,
}

// ---------------------------------------------------------------------------
// Function pointer types
// ---------------------------------------------------------------------------

type FnGetInterfaceVK = unsafe extern "C" fn(
    out_interface: *mut FfxInterface,
    device: FfxDevice,
    scratch_buffer: *mut c_void,
    scratch_buffer_size: usize,
    max_contexts: u32,
) -> FfxErrorCode;

type FnFrameInterpolationContextCreate = unsafe extern "C" fn(
    context: *mut FfxFrameInterpolationContext,
    desc: *const FfxFrameInterpolationContextDescription,
) -> FfxErrorCode;

type FnFrameInterpolationContextDestroy =
    unsafe extern "C" fn(context: *mut FfxFrameInterpolationContext) -> FfxErrorCode;

type FnFrameInterpolationGetSharedResourceDescriptions = unsafe extern "C" fn(
    context: *const FfxFrameInterpolationContext,
    desc: *mut FfxFrameInterpolationSharedResourceDescriptions,
) -> FfxErrorCode;

type FnFrameInterpolationPrepare = unsafe extern "C" fn(
    context: *mut FfxFrameInterpolationContext,
    desc: *const FfxFrameInterpolationPrepareDescription,
) -> FfxErrorCode;

type FnFrameInterpolationDispatch = unsafe extern "C" fn(
    context: *mut FfxFrameInterpolationContext,
    desc: *const FfxFrameInterpolationDispatchDescription,
) -> FfxErrorCode;

type FnGetScratchMemorySizeVK =
    unsafe extern "C" fn(physical_device: vk::PhysicalDevice, max_contexts: u32) -> usize;


// ---------------------------------------------------------------------------
// FFI declarations – statically linked by build.rs
// ---------------------------------------------------------------------------

pub unsafe fn ffxGetScratchMemorySizeVK(
    physical_device: ash::vk::PhysicalDevice, max_contexts: u32,
) -> usize {
    unsafe extern "C" {
        fn ffxGetScratchMemorySizeVK(
            physical_device: ash::vk::PhysicalDevice, max_contexts: u32,
        ) -> usize;
    }
    ffxGetScratchMemorySizeVK(physical_device, max_contexts)
}

pub unsafe fn ffxGetInterfaceVK(
    out_interface: *mut FfxInterface, device: FfxDevice,
    scratch_buffer: *mut std::ffi::c_void, scratch_buffer_size: usize, max_contexts: u32,
) -> FfxErrorCode {
    unsafe extern "C" {
        fn ffxGetInterfaceVK(
            out_interface: *mut FfxInterface, device: FfxDevice,
            scratch_buffer: *mut std::ffi::c_void, scratch_buffer_size: usize, max_contexts: u32,
        ) -> FfxErrorCode;
    }
    ffxGetInterfaceVK(out_interface, device, scratch_buffer, scratch_buffer_size, max_contexts)
}

pub unsafe fn ffxFrameInterpolationContextCreate(
    context: *mut FfxFrameInterpolationContext,
    desc: *const FfxFrameInterpolationContextDescription,
) -> FfxErrorCode {
    unsafe extern "C" {
        fn ffxFrameInterpolationContextCreate(
            context: *mut FfxFrameInterpolationContext,
            desc: *const FfxFrameInterpolationContextDescription,
        ) -> FfxErrorCode;
    }
    ffxFrameInterpolationContextCreate(context, desc)
}

pub unsafe fn ffxFrameInterpolationContextDestroy(
    context: *mut FfxFrameInterpolationContext,
) -> FfxErrorCode {
    unsafe extern "C" {
        fn ffxFrameInterpolationContextDestroy(
            context: *mut FfxFrameInterpolationContext,
        ) -> FfxErrorCode;
    }
    ffxFrameInterpolationContextDestroy(context)
}

pub unsafe fn ffxFrameInterpolationGetSharedResourceDescriptions(
    context: *const FfxFrameInterpolationContext,
    desc: *mut FfxFrameInterpolationSharedResourceDescriptions,
) -> FfxErrorCode {
    unsafe extern "C" {
        fn ffxFrameInterpolationGetSharedResourceDescriptions(
            context: *const FfxFrameInterpolationContext,
            desc: *mut FfxFrameInterpolationSharedResourceDescriptions,
        ) -> FfxErrorCode;
    }
    ffxFrameInterpolationGetSharedResourceDescriptions(context, desc)
}

pub unsafe fn ffxFrameInterpolationPrepare(
    context: *mut FfxFrameInterpolationContext,
    desc: *const FfxFrameInterpolationPrepareDescription,
) -> FfxErrorCode {
    unsafe extern "C" {
        fn ffxFrameInterpolationPrepare(
            context: *mut FfxFrameInterpolationContext,
            desc: *const FfxFrameInterpolationPrepareDescription,
        ) -> FfxErrorCode;
    }
    ffxFrameInterpolationPrepare(context, desc)
}

pub unsafe fn ffxFrameInterpolationDispatch(
    context: *mut FfxFrameInterpolationContext,
    desc: *const FfxFrameInterpolationDispatchDescription,
) -> FfxErrorCode {
    unsafe extern "C" {
        fn ffxFrameInterpolationDispatch(
            context: *mut FfxFrameInterpolationContext,
            desc: *const FfxFrameInterpolationDispatchDescription,
        ) -> FfxErrorCode;
    }
    ffxFrameInterpolationDispatch(context, desc)
}

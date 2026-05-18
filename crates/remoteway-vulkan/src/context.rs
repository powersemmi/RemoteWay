//! `VulkanContext`: shared instance/device/queue ownership.
//!
//! Ported from `remoteway-interpolate/src/backends/vulkan_context.rs` so a
//! single `VkDevice` can be reused across the FSR / frame-generation pipeline
//! and the Vulkan Video encode pipeline. All Vulkan handles are public so
//! downstream crates can issue `ash` calls directly.

#![allow(clippy::undocumented_unsafe_blocks)]

use std::ffi::CStr;

use ash::vk;
use ash::vk::TaggedStructure;

use crate::error::VulkanError;
use crate::video::{VideoCodec, VideoEncodeCapabilities};

bitflags::bitflags! {
    /// Capabilities a queue family must satisfy to be selected.
    ///
    /// Encoded as bitflags so callers can request a *single* queue family that
    /// covers multiple workloads (e.g. compute + transfer on the same family),
    /// avoiding cross-family synchronisation when the hardware allows it.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct QueueCapabilities: u32 {
        const COMPUTE      = 0b0000_0001;
        const TRANSFER     = 0b0000_0010;
        const VIDEO_ENCODE = 0b0000_0100;
    }
}

impl QueueCapabilities {
    fn to_vk_queue_flags(self) -> vk::QueueFlags {
        let mut flags = vk::QueueFlags::empty();
        if self.contains(Self::COMPUTE) {
            flags |= vk::QueueFlags::COMPUTE;
        }
        if self.contains(Self::TRANSFER) {
            flags |= vk::QueueFlags::TRANSFER;
        }
        if self.contains(Self::VIDEO_ENCODE) {
            flags |= vk::QueueFlags::VIDEO_ENCODE_KHR;
        }
        flags
    }
}

/// Description of queues the caller wants the context to expose.
#[derive(Debug, Clone)]
pub struct QueueRequest {
    /// Each entry is a distinct queue family the context must locate. The
    /// solver tries to satisfy entries with as few distinct families as
    /// possible (a single family covering multiple entries is preferred).
    pub queues: Vec<QueueCapabilities>,
    /// Video codec extensions to enable on the device. Only meaningful when
    /// at least one queue request includes `VIDEO_ENCODE`.
    pub video_codecs: Vec<VideoCodec>,
}

impl QueueRequest {
    #[must_use]
    pub fn compute_only() -> Self {
        Self {
            queues: vec![QueueCapabilities::COMPUTE | QueueCapabilities::TRANSFER],
            video_codecs: Vec::new(),
        }
    }

    #[must_use]
    pub fn compute_and_encode(codec: VideoCodec) -> Self {
        Self {
            queues: vec![
                QueueCapabilities::COMPUTE | QueueCapabilities::TRANSFER,
                QueueCapabilities::VIDEO_ENCODE,
            ],
            video_codecs: vec![codec],
        }
    }
}

/// Owns the Vulkan instance, device, and queues.
///
/// **Thread safety:** all Vulkan handles are opaque integers that are sound to
/// share across threads as long as the caller serialises mutations on the same
/// handle (`vkQueueSubmit` on the same queue, recording into the same command
/// buffer, etc.). Callers typically wrap a context in `Arc<Mutex<_>>` per
/// backend; the helpers on this type assume that synchronisation.
pub struct VulkanContext {
    pub _entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,

    /// Primary compute queue (always present).
    pub compute_queue: vk::Queue,
    /// Queue family index for `compute_queue`.
    pub queue_family: u32,
    /// Command pool bound to `queue_family`.
    pub command_pool: vk::CommandPool,

    /// Video encode queue, populated when the [`QueueRequest`] included
    /// `VIDEO_ENCODE`. `None` for compute-only contexts.
    pub video_encode_queue: Option<vk::Queue>,
    /// Queue family index for `video_encode_queue`, when present.
    pub video_encode_queue_family: Option<u32>,

    pub vendor_id: u32,
    pub device_name: String,
    pub api_version: u32,
}

// SAFETY: Vulkan handles are opaque integers; concrete safety relies on the
// caller serialising mutating calls on the same handle (Arc<Mutex<>> wrapping
// in each backend).
unsafe impl Send for VulkanContext {}
// SAFETY: Methods on this struct take `&self`; the underlying `ash::Device`
// also only takes `&self`. Interior mutation is the caller's responsibility.
unsafe impl Sync for VulkanContext {}

impl VulkanContext {
    /// Backwards-compatible constructor used by the interpolate crate.
    ///
    /// Selects a single queue family with `COMPUTE` capability and enables
    /// the requested device extensions if available.
    pub fn new(extensions: &[&CStr]) -> Result<Self, VulkanError> {
        Self::with_request(&QueueRequest::compute_only(), extensions)
    }

    /// Constructor that locates queue families satisfying `request`.
    ///
    /// Each entry in `request.queues` is satisfied by a distinct queue
    /// family (preferring fewer families when one covers multiple entries).
    /// The encode crate uses this to obtain both a compute and a video
    /// encode queue from the same device.
    pub fn with_request(
        request: &QueueRequest,
        extensions: &[&CStr],
    ) -> Result<Self, VulkanError> {
        if request.queues.is_empty() {
            return Err(VulkanError::NoSuitableQueue(
                "QueueRequest must include at least one queue".into(),
            ));
        }

        let entry = unsafe { ash::Entry::load() }
            .map_err(|e| VulkanError::LoaderFailed(format!("{e}")))?;

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"remoteway")
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(c"remoteway")
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::make_api_version(0, 1, 3, 0));

        // Opt-in validation layer when REMOTEWAY_VK_VALIDATION=1 is set.
        // Useful for diagnosing video-encode-time DEVICE_LOST issues.
        let want_validation = std::env::var("REMOTEWAY_VK_VALIDATION")
            .ok()
            .as_deref() == Some("1");
        let layer_names: Vec<*const i8> = if want_validation {
            vec![c"VK_LAYER_KHRONOS_validation".as_ptr() as *const _]
        } else {
            Vec::new()
        };

        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layer_names);
        let instance = unsafe { entry.create_instance(&instance_info, None) }
            .map_err(|e| VulkanError::InitFailed(format!("Vulkan instance: {e}")))?;

        let physical_devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|e| VulkanError::InitFailed(format!("enumerate devices: {e}")))?;

        if physical_devices.is_empty() {
            unsafe { instance.destroy_instance(None) };
            return Err(VulkanError::NoDevice);
        }

        let physical_device = physical_devices
            .iter()
            .find(|&&pd| {
                let props = unsafe { instance.get_physical_device_properties(pd) };
                props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
            })
            .copied()
            .unwrap_or(physical_devices[0]);

        let props = unsafe { instance.get_physical_device_properties(physical_device) };
        let device_name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let vendor_id = props.vendor_id;
        let api_version = props.api_version;

        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

        // Resolve each requested queue capability set to a queue family
        // index. We greedily pick the family that *exactly* matches the
        // requested flags when possible (avoids picking a generic family
        // that includes more flags than asked for — preserves separate
        // dedicated video-encode queue when the hardware exposes one).
        let mut selected: Vec<(QueueCapabilities, u32)> = Vec::with_capacity(request.queues.len());
        for &caps in &request.queues {
            let wanted = caps.to_vk_queue_flags();
            let family_idx = queue_families
                .iter()
                .enumerate()
                .find(|(_, qf)| qf.queue_flags.contains(wanted))
                .map(|(i, _)| i as u32)
                .ok_or_else(|| VulkanError::NoSuitableQueue(format!("{caps:?}")))?;
            selected.push((caps, family_idx));
        }

        // Build distinct DeviceQueueCreateInfo entries (one per unique family).
        let mut unique_families: Vec<u32> = selected.iter().map(|(_, f)| *f).collect();
        unique_families.sort_unstable();
        unique_families.dedup();
        let queue_priority = [1.0f32];
        let queue_infos: Vec<vk::DeviceQueueCreateInfo<'_>> = unique_families
            .iter()
            .map(|&family| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(family)
                    .queue_priorities(&queue_priority)
            })
            .collect();

        // Collect extensions: caller-supplied + codec encode extensions +
        // video-encode-queue extension when a VIDEO_ENCODE queue was requested.
        let mut needed_ext_cstrs: Vec<&CStr> = extensions.to_vec();
        let wants_encode = request
            .queues
            .iter()
            .any(|c| c.contains(QueueCapabilities::VIDEO_ENCODE));
        if wants_encode {
            needed_ext_cstrs.push(c"VK_KHR_video_queue");
            needed_ext_cstrs.push(c"VK_KHR_video_encode_queue");
            for codec in &request.video_codecs {
                needed_ext_cstrs.push(codec.encode_extension());
            }
        }

        let available_exts =
            unsafe { instance.enumerate_device_extension_properties(physical_device) }
                .unwrap_or_default();
        let ext_ptrs: Vec<*const i8> = needed_ext_cstrs
            .iter()
            .filter(|&&ext| {
                available_exts.iter().any(|ae| {
                    let name = unsafe { CStr::from_ptr(ae.extension_name.as_ptr()) };
                    name == ext
                })
            })
            .map(|ext| ext.as_ptr())
            .collect();

        // AV1 encode requires PhysicalDeviceVideoEncodeAV1FeaturesKHR.video_encode_av1
        // to be explicitly enabled at device creation. H.264/H.265 don't gate
        // device creation on a dedicated feature struct.
        let want_av1 = request.video_codecs.contains(&VideoCodec::Av1);
        let mut av1_feature =
            vk::PhysicalDeviceVideoEncodeAV1FeaturesKHR::default().video_encode_av1(want_av1);

        let mut device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&ext_ptrs);
        if want_av1 {
            device_info = device_info.push(&mut av1_feature);
        }

        let device = unsafe { instance.create_device(physical_device, &device_info, None) }
            .map_err(|e| {
                unsafe { instance.destroy_instance(None) };
                VulkanError::InitFailed(format!("Vulkan device: {e}"))
            })?;

        // Pull queues out for compute and (optionally) encode. By contract
        // the first COMPUTE-bearing request becomes `compute_queue`.
        let (_, compute_family) = selected
            .iter()
            .find(|(c, _)| c.contains(QueueCapabilities::COMPUTE))
            .copied()
            .ok_or_else(|| VulkanError::NoSuitableQueue("no COMPUTE queue requested".into()))?;
        let compute_queue = unsafe { device.get_device_queue(compute_family, 0) };

        let (video_encode_queue, video_encode_family) = selected
            .iter()
            .find(|(c, _)| c.contains(QueueCapabilities::VIDEO_ENCODE))
            .map(|(_, family)| {
                let q = unsafe { device.get_device_queue(*family, 0) };
                (Some(q), Some(*family))
            })
            .unwrap_or((None, None));

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(compute_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.create_command_pool(&pool_info, None) }
            .map_err(|e| {
                unsafe {
                    device.destroy_device(None);
                    instance.destroy_instance(None);
                }
                VulkanError::InitFailed(format!("command pool: {e}"))
            })?;

        Ok(Self {
            _entry: entry,
            instance,
            physical_device,
            device,
            compute_queue,
            queue_family: compute_family,
            command_pool,
            video_encode_queue,
            video_encode_queue_family: video_encode_family,
            vendor_id,
            device_name,
            api_version,
        })
    }

    /// Can we create a Vulkan instance and find any compute GPU?
    #[must_use]
    pub fn is_vulkan_available() -> bool {
        Self::new(&[]).is_ok()
    }

    /// NVIDIA-specific: `VK_NV_optical_flow` available.
    #[must_use]
    pub fn probe_nvidia_optical_flow() -> bool {
        let ctx = match Self::new(&[]) {
            Ok(ctx) => ctx,
            Err(_) => return false,
        };
        if ctx.vendor_id != 0x10DE {
            return false;
        }
        let exts = unsafe {
            ctx.instance
                .enumerate_device_extension_properties(ctx.physical_device)
        }
        .unwrap_or_default();
        exts.iter().any(|e| {
            let name = unsafe { CStr::from_ptr(e.extension_name.as_ptr()) };
            name == c"VK_NV_optical_flow"
        })
    }

    /// AMD-specific: RDNA3 (RX 7000) or newer.
    #[must_use]
    pub fn probe_rdna3_plus() -> bool {
        let ctx = match Self::new(&[]) {
            Ok(ctx) => ctx,
            Err(_) => return false,
        };
        if ctx.vendor_id != 0x1002 {
            return false;
        }
        let name = ctx.device_name.to_lowercase();
        name.contains("rx 7")
            || name.contains("rx 9")
            || name.contains("rdna 3")
            || name.contains("rdna 4")
            || name.contains("radeon 7")
            || name.contains("radeon 9")
    }

    /// Probes video encode capabilities for the given codec on the selected
    /// physical device.
    ///
    /// Wraps `vkGetPhysicalDeviceVideoCapabilitiesKHR` with a codec-appropriate
    /// `VkVideoProfileInfoKHR` (Main profile, 4:2:0 chroma, 8-bit per channel —
    /// the realistic baseline for low-latency desktop streaming). The caller
    /// validates user-supplied `EncodeParams` against the returned struct.
    ///
    /// Errors:
    /// - [`VulkanError::Call`] with the underlying `vk::Result` if the driver
    ///   does not support the requested codec at the requested profile.
    pub fn probe_video_encode_capabilities(
        &self,
        codec: VideoCodec,
    ) -> Result<VideoEncodeCapabilities, VulkanError> {
        match codec {
            VideoCodec::H264 => self.probe_h264(),
            VideoCodec::H265 => self.probe_h265(),
            VideoCodec::Av1 => self.probe_av1(),
        }
    }

    fn probe_h264(&self) -> Result<VideoEncodeCapabilities, VulkanError> {
        let mut h264_profile = vk::VideoEncodeH264ProfileInfoKHR::default()
            .std_profile_idc(vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_MAIN);
        let profile = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(VideoCodec::H264.codec_operation())
            .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
            .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .push(&mut h264_profile);

        let mut h264_caps = vk::VideoEncodeH264CapabilitiesKHR::default();
        let mut encode_caps = vk::VideoEncodeCapabilitiesKHR::default();
        let mut caps = vk::VideoCapabilitiesKHR::default()
            .push(&mut encode_caps)
            .push(&mut h264_caps);

        self.fetch_caps(VideoCodec::H264, &profile, &mut caps)
    }

    fn probe_h265(&self) -> Result<VideoEncodeCapabilities, VulkanError> {
        let mut h265_profile = vk::VideoEncodeH265ProfileInfoKHR::default()
            .std_profile_idc(vk::native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN);
        let profile = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(VideoCodec::H265.codec_operation())
            .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
            .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .push(&mut h265_profile);

        let mut h265_caps = vk::VideoEncodeH265CapabilitiesKHR::default();
        let mut encode_caps = vk::VideoEncodeCapabilitiesKHR::default();
        let mut caps = vk::VideoCapabilitiesKHR::default()
            .push(&mut encode_caps)
            .push(&mut h265_caps);

        self.fetch_caps(VideoCodec::H265, &profile, &mut caps)
    }

    fn probe_av1(&self) -> Result<VideoEncodeCapabilities, VulkanError> {
        let mut av1_profile = vk::VideoEncodeAV1ProfileInfoKHR::default()
            .std_profile(vk::native::StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_MAIN);
        let profile = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(VideoCodec::Av1.codec_operation())
            .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
            .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .push(&mut av1_profile);

        let mut av1_caps = vk::VideoEncodeAV1CapabilitiesKHR::default();
        let mut encode_caps = vk::VideoEncodeCapabilitiesKHR::default();
        let mut caps = vk::VideoCapabilitiesKHR::default()
            .push(&mut encode_caps)
            .push(&mut av1_caps);

        self.fetch_caps(VideoCodec::Av1, &profile, &mut caps)
    }

    /// Issues the actual capabilities query and assembles the public `VideoEncodeCapabilities`.
    ///
    /// Per the Vulkan spec, the driver writes into every chained extension struct found in
    /// `caps.p_next`. We walk the chain after the call to pull `rate_control_modes` out of
    /// the `VkVideoEncodeCapabilitiesKHR` struct the caller chained in.
    fn fetch_caps(
        &self,
        codec: VideoCodec,
        profile: &vk::VideoProfileInfoKHR<'_>,
        caps: &mut vk::VideoCapabilitiesKHR<'_>,
    ) -> Result<VideoEncodeCapabilities, VulkanError> {
        let video_queue_fn =
            ash::khr::video_queue::Instance::load(&self._entry, &self.instance);

        unsafe {
            video_queue_fn.get_physical_device_video_capabilities(
                self.physical_device,
                profile,
                caps,
            )
        }
        .map_err(|e| VulkanError::call("get_physical_device_video_capabilities", e))?;

        // After the call, the driver has written into the chained encode_caps struct.
        // We pull the encode-specific bits back out by walking the p_next chain.
        let rate_control_modes = unsafe { Self::extract_encode_caps(caps) };

        let exts = unsafe {
            self.instance
                .enumerate_device_extension_properties(self.physical_device)
        }
        .unwrap_or_default();
        let has_ext = |name: &CStr| {
            exts.iter()
                .any(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) } == name)
        };

        Ok(VideoEncodeCapabilities {
            codec,
            max_coded_extent: (caps.max_coded_extent.width, caps.max_coded_extent.height),
            min_coded_extent: (caps.min_coded_extent.width, caps.min_coded_extent.height),
            picture_access_granularity: (
                caps.picture_access_granularity.width,
                caps.picture_access_granularity.height,
            ),
            max_dpb_slots: caps.max_dpb_slots,
            max_active_reference_pictures: caps.max_active_reference_pictures,
            rate_control_modes,
            supports_intra_refresh: has_ext(c"VK_KHR_video_encode_intra_refresh"),
            supports_quantization_map: has_ext(c"VK_KHR_video_encode_quantization_map"),
        })
    }

    /// Walks the `p_next` chain of a populated `VideoCapabilitiesKHR` to find the
    /// `VkVideoEncodeCapabilitiesKHR` struct and read its `rate_control_modes` field.
    ///
    /// Safety: the chain must have been populated by `vkGetPhysicalDeviceVideoCapabilitiesKHR`
    /// with a `VkVideoEncodeCapabilitiesKHR` chained in.
    unsafe fn extract_encode_caps(
        caps: &vk::VideoCapabilitiesKHR<'_>,
    ) -> vk::VideoEncodeRateControlModeFlagsKHR {
        let mut p = caps.p_next as *const vk::BaseOutStructure<'_>;
        while !p.is_null() {
            // SAFETY: every node in the p_next chain begins with the BaseOutStructure layout
            // (Vulkan guarantees this for any pNext-chainable struct). The chain was set up
            // by us via `push(&mut encode_caps)` so every pointer either targets a struct we
            // own that is still alive or is null.
            let node = unsafe { &*p };
            if node.s_type == vk::StructureType::VIDEO_ENCODE_CAPABILITIES_KHR {
                let encode = p.cast::<vk::VideoEncodeCapabilitiesKHR<'_>>();
                // SAFETY: the s_type tag matches the cast target type, the struct is alive,
                // and the driver populated it during `get_physical_device_video_capabilities`.
                return unsafe { (*encode).rate_control_modes };
            }
            p = node.p_next as *const _;
        }
        vk::VideoEncodeRateControlModeFlagsKHR::empty()
    }

    /// Allocate a host-visible buffer, preferring cached memory for fast CPU reads.
    pub fn create_host_buffer(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> Result<(vk::Buffer, vk::DeviceMemory), VulkanError> {
        let cached = vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT
            | vk::MemoryPropertyFlags::HOST_CACHED;
        let uncached =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        self.create_buffer(size, usage, cached)
            .or_else(|_| self.create_buffer(size, usage, uncached))
    }

    /// Allocate a buffer with the given size, usage and memory flags.
    pub fn create_buffer(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_flags: vk::MemoryPropertyFlags,
    ) -> Result<(vk::Buffer, vk::DeviceMemory), VulkanError> {
        let buf_info = vk::BufferCreateInfo::default().size(size).usage(usage);

        let buffer = unsafe { self.device.create_buffer(&buf_info, None) }
            .map_err(|e| VulkanError::Allocation(format!("create buffer: {e}")))?;

        let mem_req = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let mem_props = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        };
        let mem_type_index = (0..mem_props.memory_type_count)
            .find(|&i| {
                let type_bits = mem_req.memory_type_bits & (1 << i);
                let prop_match = mem_props.memory_types[i as usize]
                    .property_flags
                    .contains(memory_flags);
                type_bits != 0 && prop_match
            })
            .ok_or_else(|| {
                unsafe { self.device.destroy_buffer(buffer, None) };
                VulkanError::Allocation("no suitable memory type".into())
            })?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_req.size)
            .memory_type_index(mem_type_index);

        let memory = unsafe { self.device.allocate_memory(&alloc_info, None) }.map_err(|e| {
            unsafe { self.device.destroy_buffer(buffer, None) };
            VulkanError::Allocation(format!("allocate memory: {e}"))
        })?;

        unsafe { self.device.bind_buffer_memory(buffer, memory, 0) }.map_err(|e| {
            unsafe {
                self.device.free_memory(memory, None);
                self.device.destroy_buffer(buffer, None);
            }
            VulkanError::Allocation(format!("bind memory: {e}"))
        })?;

        Ok((buffer, memory))
    }

    /// Upload data to a host-visible buffer.
    pub fn upload_to_buffer(
        &self,
        memory: vk::DeviceMemory,
        data: &[u8],
    ) -> Result<(), VulkanError> {
        unsafe {
            let ptr = self
                .device
                .map_memory(memory, 0, data.len() as u64, vk::MemoryMapFlags::empty())
                .map_err(|e| VulkanError::Allocation(format!("map memory: {e}")))?;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
            self.device.unmap_memory(memory);
        }
        Ok(())
    }

    /// Read data from a host-visible buffer.
    pub fn read_from_buffer(
        &self,
        memory: vk::DeviceMemory,
        size: usize,
    ) -> Result<Vec<u8>, VulkanError> {
        unsafe {
            let ptr = self
                .device
                .map_memory(memory, 0, size as u64, vk::MemoryMapFlags::empty())
                .map_err(|e| VulkanError::Allocation(format!("map memory: {e}")))?;
            let mut data = vec![0u8; size];
            std::ptr::copy_nonoverlapping(ptr as *const u8, data.as_mut_ptr(), size);
            self.device.unmap_memory(memory);
            Ok(data)
        }
    }

    /// Submit a command buffer on the compute queue and wait for completion.
    pub fn submit_and_wait(&self, cmd: vk::CommandBuffer) -> Result<(), VulkanError> {
        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe { self.device.create_fence(&fence_info, None) }
            .map_err(|e| VulkanError::call("create_fence", e))?;

        let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));

        let result = unsafe {
            self.device
                .queue_submit(self.compute_queue, &[submit_info], fence)
                .map_err(|e| VulkanError::call("queue_submit", e))
                .and_then(|()| {
                    self.device
                        .wait_for_fences(&[fence], true, u64::MAX)
                        .map_err(|e| VulkanError::call("wait_for_fences", e))
                })
        };

        unsafe { self.device.destroy_fence(fence, None) };

        result
    }

    /// Allocate a single command buffer from the pool.
    pub fn allocate_command_buffer(&self) -> Result<vk::CommandBuffer, VulkanError> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let cmd = unsafe { self.device.allocate_command_buffers(&alloc_info) }
            .map_err(|e| VulkanError::call("allocate_command_buffers", e))?;
        Ok(cmd[0])
    }

    /// Create a device-local 2D RGBA8 image of the given dimensions.
    pub fn create_image(
        &self,
        width: u32,
        height: u32,
        usage: vk::ImageUsageFlags,
    ) -> Result<(vk::Image, vk::DeviceMemory), VulkanError> {
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

        let image = unsafe { self.device.create_image(&img_info, None) }
            .map_err(|e| VulkanError::Allocation(format!("create image: {e}")))?;

        let mem_req = unsafe { self.device.get_image_memory_requirements(image) };
        let mem_props = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        };
        let mem_type_index = (0..mem_props.memory_type_count)
            .find(|&i| {
                mem_req.memory_type_bits & (1 << i) != 0
                    && mem_props.memory_types[i as usize]
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            })
            .ok_or_else(|| {
                unsafe { self.device.destroy_image(image, None) };
                VulkanError::Allocation("no device-local memory for image".into())
            })?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_req.size)
            .memory_type_index(mem_type_index);
        let mem = unsafe { self.device.allocate_memory(&alloc_info, None) }.map_err(|e| {
            unsafe { self.device.destroy_image(image, None) };
            VulkanError::Allocation(format!("alloc image mem: {e}"))
        })?;
        unsafe { self.device.bind_image_memory(image, mem, 0) }.map_err(|e| {
            unsafe {
                self.device.free_memory(mem, None);
                self.device.destroy_image(image, None);
            }
            VulkanError::Allocation(format!("bind image mem: {e}"))
        })?;

        Ok((image, mem))
    }

    /// Issue an image layout transition barrier.
    pub fn cmd_image_barrier(
        &self,
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
            self.device.cmd_pipeline_barrier(
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

    /// GPU-accelerated bilinear upscale using `vkCmdBlitImage`.
    pub fn upscale_blit(
        &self,
        src: &[u8],
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
    ) -> Result<Vec<u8>, VulkanError> {
        let src_size = u64::from(src_w) * u64::from(src_h) * 4;
        let dst_size = u64::from(dst_w) * u64::from(dst_h) * 4;

        let (staging_buf, staging_mem) = self.create_buffer(
            src_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        self.upload_to_buffer(staging_mem, src)?;

        let (src_image, src_image_mem) = self.create_image(
            src_w,
            src_h,
            vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
        )?;
        let (dst_image, dst_image_mem) = self.create_image(
            dst_w,
            dst_h,
            vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
        )?;

        let (readback_buf, readback_mem) = self.create_buffer(
            dst_size,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let cmd = self.allocate_command_buffer()?;
        let begin_info = vk::CommandBufferBeginInfo::default();
        unsafe { self.device.begin_command_buffer(cmd, &begin_info) }
            .map_err(|e| VulkanError::call("begin_command_buffer", e))?;

        self.cmd_image_barrier(
            cmd,
            src_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
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
            .image_extent(vk::Extent3D::default().width(src_w).height(src_h).depth(1));
        unsafe {
            self.device.cmd_copy_buffer_to_image(
                cmd,
                staging_buf,
                src_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&copy_region),
            );
        }

        self.cmd_image_barrier(
            cmd,
            src_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        );
        self.cmd_image_barrier(
            cmd,
            dst_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );

        let blit = vk::ImageBlit::default()
            .src_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .src_offsets([
                vk::Offset3D::default(),
                vk::Offset3D::default().x(src_w as i32).y(src_h as i32).z(1),
            ])
            .dst_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .dst_offsets([
                vk::Offset3D::default(),
                vk::Offset3D::default().x(dst_w as i32).y(dst_h as i32).z(1),
            ]);
        unsafe {
            self.device.cmd_blit_image(
                cmd,
                src_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&blit),
                vk::Filter::LINEAR,
            );
        }

        self.cmd_image_barrier(
            cmd,
            dst_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
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
        unsafe {
            self.device.cmd_copy_image_to_buffer(
                cmd,
                dst_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                readback_buf,
                std::slice::from_ref(&copy_region),
            );
        }

        unsafe { self.device.end_command_buffer(cmd) }
            .map_err(|e| VulkanError::call("end_command_buffer", e))?;

        self.submit_and_wait(cmd)?;
        let result = self.read_from_buffer(readback_mem, dst_size as usize)?;

        unsafe {
            self.device.destroy_buffer(staging_buf, None);
            self.device.free_memory(staging_mem, None);
            self.device.destroy_image(src_image, None);
            self.device.free_memory(src_image_mem, None);
            self.device.destroy_image(dst_image, None);
            self.device.free_memory(dst_image_mem, None);
            self.device.destroy_buffer(readback_buf, None);
            self.device.free_memory(readback_mem, None);
            self.device.free_command_buffers(self.command_pool, &[cmd]);
        }

        Ok(result)
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vulkan_context_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<VulkanContext>();
    }

    #[test]
    fn queue_request_compute_only_has_compute() {
        let req = QueueRequest::compute_only();
        assert_eq!(req.queues.len(), 1);
        assert!(req.queues[0].contains(QueueCapabilities::COMPUTE));
        assert!(req.video_codecs.is_empty());
    }

    #[test]
    fn queue_request_compute_and_encode_lists_both() {
        let req = QueueRequest::compute_and_encode(VideoCodec::H265);
        assert_eq!(req.queues.len(), 2);
        assert!(req.queues.iter().any(|c| c.contains(QueueCapabilities::COMPUTE)));
        assert!(req.queues.iter().any(|c| c.contains(QueueCapabilities::VIDEO_ENCODE)));
        assert_eq!(req.video_codecs, vec![VideoCodec::H265]);
    }

    #[test]
    fn queue_capabilities_to_vk_includes_video_encode_bit() {
        let caps = QueueCapabilities::VIDEO_ENCODE;
        let vk_flags = caps.to_vk_queue_flags();
        assert!(vk_flags.contains(vk::QueueFlags::VIDEO_ENCODE_KHR));
    }

    #[test]
    #[ignore] // requires Vulkan runtime
    fn vulkan_context_creation() {
        let ctx = VulkanContext::new(&[]);
        assert!(ctx.is_ok(), "failed: {:?}", ctx.err());
        let ctx = ctx.unwrap();
        assert!(!ctx.device_name.is_empty());
        assert!(ctx.vendor_id > 0);
    }

    #[test]
    #[ignore] // requires Vulkan runtime + a video-encode-capable device
    fn vulkan_context_with_video_encode() {
        let ctx = VulkanContext::with_request(
            &QueueRequest::compute_and_encode(VideoCodec::H265),
            &[],
        );
        let ctx = ctx.expect("failed to create encode context");
        assert!(ctx.video_encode_queue.is_some(), "no video encode queue");
        assert!(ctx.video_encode_queue_family.is_some());
    }

    #[test]
    #[ignore] // requires Vulkan runtime + a video-encode-capable device
    fn probe_h265_capabilities() {
        let ctx = VulkanContext::with_request(
            &QueueRequest::compute_and_encode(VideoCodec::H265),
            &[],
        )
        .expect("encode context");
        let caps = ctx
            .probe_video_encode_capabilities(VideoCodec::H265)
            .expect("h265 caps");
        assert_eq!(caps.codec, VideoCodec::H265);
        assert!(
            caps.max_coded_extent.0 >= 1920 && caps.max_coded_extent.1 >= 1080,
            "max extent too small: {:?}",
            caps.max_coded_extent
        );
        assert!(caps.max_dpb_slots >= 2, "need 2 DPB slots for low-latency");
    }

    #[test]
    #[ignore] // requires Vulkan runtime + a video-encode-capable device
    fn probe_h264_capabilities() {
        let ctx = VulkanContext::with_request(
            &QueueRequest::compute_and_encode(VideoCodec::H264),
            &[],
        )
        .expect("encode context");
        let caps = ctx
            .probe_video_encode_capabilities(VideoCodec::H264)
            .expect("h264 caps");
        assert_eq!(caps.codec, VideoCodec::H264);
        assert!(caps.max_coded_extent.0 >= 1920);
    }

    #[test]
    #[ignore] // requires Vulkan runtime + a video-encode-capable device + AV1 support
    fn probe_av1_capabilities() {
        let ctx = VulkanContext::with_request(
            &QueueRequest::compute_and_encode(VideoCodec::Av1),
            &[],
        )
        .expect("encode context");
        let caps = ctx
            .probe_video_encode_capabilities(VideoCodec::Av1)
            .expect("av1 caps");
        assert_eq!(caps.codec, VideoCodec::Av1);
        assert!(caps.max_coded_extent.0 >= 1920);
    }

    #[test]
    #[ignore] // requires Vulkan runtime
    fn vulkan_buffer_roundtrip() {
        let ctx = VulkanContext::new(&[]).unwrap();
        let data = vec![0xABu8; 1024];
        let (buf, mem) = ctx
            .create_buffer(
                1024,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .unwrap();
        ctx.upload_to_buffer(mem, &data).unwrap();
        let readback = ctx.read_from_buffer(mem, 1024).unwrap();
        assert_eq!(data, readback);

        unsafe {
            ctx.device.destroy_buffer(buf, None);
            ctx.device.free_memory(mem, None);
        }
    }
}

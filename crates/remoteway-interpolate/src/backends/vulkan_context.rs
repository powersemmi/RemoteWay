use std::ffi::CStr;

use ash::vk;

use crate::error::InterpolateError;

/// Shared Vulkan context for ash-based interpolation backends (FSR2, FSR3, NVIDIA-OF).
///
/// Manages Vulkan instance, device, compute queue, and command pool.
/// All Vulkan calls must be externally synchronized (via Mutex in each backend).
#[allow(dead_code)] // Fields used across different feature combinations.
pub(crate) struct VulkanContext {
    _entry: ash::Entry,
    pub(crate) instance: ash::Instance,
    pub(crate) physical_device: vk::PhysicalDevice,
    pub(crate) device: ash::Device,
    pub(crate) compute_queue: vk::Queue,
    pub(crate) queue_family: u32,
    pub(crate) command_pool: vk::CommandPool,
    pub(crate) vendor_id: u32,
    pub(crate) device_name: String,
    pub(crate) api_version: u32,
}

// Safety: All Vulkan handles are opaque integers (u64/usize) and are thread-safe
// when externally synchronized. Each backend wraps VulkanContext in Arc<Mutex<>>,
// ensuring all Vulkan calls are serialized.
unsafe impl Send for VulkanContext {}
// SAFETY: All Vulkan handles in VulkanContext are thread-safe for shared (&T)
// access because ash methods take &self on handles. Interior mutability is not
// used. The Arc<Mutex<>> wrappers in backends serialize all mutable GPU state
// changes, so shared access to the VulkanContext across threads is sound.
unsafe impl Sync for VulkanContext {}

impl VulkanContext {
    /// Create a new Vulkan context with optional device extensions.
    pub(crate) fn new(extensions: &[&CStr]) -> Result<Self, InterpolateError> {
        // SAFETY: ash::Entry::load() loads the Vulkan loader library dynamically.
        // The entry is only used to create an instance; no Vulkan state is
        // modified before instance creation.
        let entry = unsafe { ash::Entry::load() }
            .map_err(|e| InterpolateError::InitFailed(format!("Vulkan loader: {e}")))?;

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"remoteway-interpolate")
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(c"remoteway")
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::make_api_version(0, 1, 3, 0));

        let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);

        // SAFETY: The entry was loaded successfully above. No validation layers
        // or instance extensions are needed for headless compute. The
        // ApplicationInfo and CreateInfo are statically initialized and valid.
        let instance = unsafe { entry.create_instance(&instance_info, None) }
            .map_err(|e| InterpolateError::InitFailed(format!("Vulkan instance: {e}")))?;

        // Enumerate physical devices.
        // SAFETY: The instance is valid (just created above). This call only
        // reads physical device handles from the Vulkan implementation.
        let physical_devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|e| InterpolateError::InitFailed(format!("enumerate devices: {e}")))?;

        if physical_devices.is_empty() {
            // SAFETY: The instance is valid and no other Vulkan objects
            // (device, command pool, etc.) have been created yet, so
            // destroying the instance directly is safe.
            unsafe { instance.destroy_instance(None) };
            return Err(InterpolateError::InitFailed(
                "no Vulkan physical devices".into(),
            ));
        }

        // Prefer discrete GPU.
        let physical_device = physical_devices
            .iter()
            .find(|&&pd| {
                // SAFETY: pd is a valid physical device handle from
                // enumerate_physical_devices. This call only reads static
                // properties from the driver.
                let props = unsafe { instance.get_physical_device_properties(pd) };
                props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
            })
            .copied()
            .unwrap_or(physical_devices[0]);

        // SAFETY: physical_device was selected from enumeration above and is
        // a valid handle. This reads static device properties from the driver.
        let props = unsafe { instance.get_physical_device_properties(physical_device) };
        // SAFETY: props.device_name is a fixed-size null-terminated C string
        // populated by get_physical_device_properties. CStr::from_ptr reads
        // only up to the null terminator within the fixed array bounds.
        let device_name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let vendor_id = props.vendor_id;
        let api_version = props.api_version;

        // Find compute queue family.
        // SAFETY: physical_device is a valid handle from enumeration. This call
        // reads static queue family properties from the driver.
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let queue_family = queue_families
            .iter()
            .enumerate()
            .find(|(_, qf)| qf.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|(i, _)| i as u32)
            .ok_or_else(|| {
                // SAFETY: The instance is valid and no device has been created
                // yet, so destroying the instance directly is safe.
                unsafe { instance.destroy_instance(None) };
                InterpolateError::InitFailed("no compute queue family".into())
            })?;

        // Check requested device extensions are available.
        // SAFETY: instance and physical_device are valid handles. This reads
        // the list of supported device extensions from the driver.
        let available_exts =
            unsafe { instance.enumerate_device_extension_properties(physical_device) }
                .unwrap_or_default();
        let ext_ptrs: Vec<*const i8> = extensions
            .iter()
            .filter(|&&ext| {
                available_exts.iter().any(|ae| {
                    // SAFETY: extension_name is a fixed-size null-terminated
                    // C string populated by enumerate_device_extension_properties.
                    let name = unsafe { CStr::from_ptr(ae.extension_name.as_ptr()) };
                    name == ext
                })
            })
            .map(|ext| ext.as_ptr())
            .collect();

        // If not all requested extensions were found, that's OK — the caller
        // will check capabilities. We just enable what's available.

        let queue_priority = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&queue_priority);

        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_extension_names(&ext_ptrs);

        // SAFETY: physical_device is a valid handle, device_info references
        // valid queue create info and extension name pointers with static
        // lifetimes, and no allocation callbacks are needed.
        let device = unsafe { instance.create_device(physical_device, &device_info, None) }
            .map_err(|e| {
                // SAFETY: The instance is valid and no logical device was
                // created, so destroying the instance is safe.
                unsafe { instance.destroy_instance(None) };
                InterpolateError::InitFailed(format!("Vulkan device: {e}"))
            })?;

        // SAFETY: device is a valid handle (just created above) and
        // queue_family is a valid index obtained from enumeration.
        let compute_queue = unsafe { device.get_device_queue(queue_family, 0) };

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        // SAFETY: device is a valid handle, pool_info is correctly
        // initialized with the compute queue family index.
        let command_pool =
            unsafe { device.create_command_pool(&pool_info, None) }.map_err(|e| {
                // SAFETY: Both device and instance are valid handles.
                // Destroying them in reverse creation order is correct.
                unsafe {
                    device.destroy_device(None);
                    instance.destroy_instance(None);
                }
                InterpolateError::InitFailed(format!("command pool: {e}"))
            })?;

        Ok(Self {
            _entry: entry,
            instance,
            physical_device,
            device,
            compute_queue,
            queue_family,
            command_pool,
            vendor_id,
            device_name,
            api_version,
        })
    }

    /// Can we create a Vulkan instance and find any compute GPU?
    #[allow(dead_code)] // Used by fsr2 feature.
    pub(crate) fn is_vulkan_available() -> bool {
        Self::new(&[]).is_ok()
    }

    /// Check if the NVIDIA `VK_NV_optical_flow` extension is available.
    #[allow(dead_code)] // Used by nvidia-of and fsr3 features.
    pub(crate) fn probe_nvidia_optical_flow() -> bool {
        let ctx = match Self::new(&[]) {
            Ok(ctx) => ctx,
            Err(_) => return false,
        };
        if ctx.vendor_id != 0x10DE {
            return false;
        }
        // SAFETY: ctx.instance and ctx.physical_device are valid handles
        // from a successfully created VulkanContext.
        let exts = unsafe {
            ctx.instance
                .enumerate_device_extension_properties(ctx.physical_device)
        }
        .unwrap_or_default();
        exts.iter().any(|e| {
            // SAFETY: extension_name is a fixed-size null-terminated
            // C string populated by enumerate_device_extension_properties.
            let name = unsafe { CStr::from_ptr(e.extension_name.as_ptr()) };
            name == c"VK_NV_optical_flow"
        })
    }

    /// Check if the device is AMD RDNA3 or newer (vendor 0x1002 + known device patterns).
    #[allow(dead_code)] // Used by fsr3 feature.
    pub(crate) fn probe_rdna3_plus() -> bool {
        let ctx = match Self::new(&[]) {
            Ok(ctx) => ctx,
            Err(_) => return false,
        };
        if ctx.vendor_id != 0x1002 {
            return false;
        }
        // RDNA3: RX 7000 series. RDNA4: RX 9000 series.
        let name = ctx.device_name.to_lowercase();
        name.contains("rx 7")
            || name.contains("rx 9")
            || name.contains("rdna 3")
            || name.contains("rdna 4")
            || name.contains("radeon 7")
            || name.contains("radeon 9")
    }

    /// Allocate a device-local buffer with the given size and usage.
    pub(crate) fn create_buffer(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_flags: vk::MemoryPropertyFlags,
    ) -> Result<(vk::Buffer, vk::DeviceMemory), InterpolateError> {
        let buf_info = vk::BufferCreateInfo::default().size(size).usage(usage);

        // SAFETY: self.device is a valid handle and buf_info is correctly
        // initialized with valid size and usage flags.
        let buffer = unsafe { self.device.create_buffer(&buf_info, None) }
            .map_err(|e| InterpolateError::InterpolateFailed(format!("create buffer: {e}")))?;

        // SAFETY: buffer is a valid handle (just created above).
        let mem_req = unsafe { self.device.get_buffer_memory_requirements(buffer) };

        // SAFETY: self.instance and self.physical_device are valid handles.
        // This reads static memory properties from the driver.
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
                // SAFETY: buffer is a valid handle created above and
                // is not currently in use by any GPU operation.
                unsafe { self.device.destroy_buffer(buffer, None) };
                InterpolateError::InterpolateFailed("no suitable memory type".into())
            })?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_req.size)
            .memory_type_index(mem_type_index);

        // SAFETY: self.device is a valid handle and alloc_info specifies
        // a valid memory type index and allocation size.
        let memory = unsafe { self.device.allocate_memory(&alloc_info, None) }.map_err(|e| {
            // SAFETY: buffer is a valid handle and not bound to any memory yet.
            unsafe { self.device.destroy_buffer(buffer, None) };
            InterpolateError::InterpolateFailed(format!("allocate memory: {e}"))
        })?;

        // SAFETY: buffer and memory are valid handles, the allocation
        // size covers the buffer requirements (checked via mem_req), and
        // the memory type is compatible with the buffer.
        unsafe { self.device.bind_buffer_memory(buffer, memory, 0) }.map_err(|e| {
            // SAFETY: memory and buffer are valid handles that were just
            // allocated/created. They are not in use by the GPU.
            unsafe {
                self.device.free_memory(memory, None);
                self.device.destroy_buffer(buffer, None);
            }
            InterpolateError::InterpolateFailed(format!("bind memory: {e}"))
        })?;

        Ok((buffer, memory))
    }

    /// Upload data to a host-visible buffer.
    pub(crate) fn upload_to_buffer(
        &self,
        memory: vk::DeviceMemory,
        data: &[u8],
    ) -> Result<(), InterpolateError> {
        // SAFETY: memory is a valid VkDeviceMemory handle allocated with
        // HOST_VISIBLE property. The mapped range [0, data.len()) is within
        // the allocation size. copy_nonoverlapping is sound because both
        // data.as_ptr() and the mapped pointer are valid for data.len()
        // bytes and they do not overlap. unmap_memory is called before
        // returning.
        unsafe {
            let ptr = self
                .device
                .map_memory(memory, 0, data.len() as u64, vk::MemoryMapFlags::empty())
                .map_err(|e| InterpolateError::InterpolateFailed(format!("map memory: {e}")))?;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
            self.device.unmap_memory(memory);
        }
        Ok(())
    }

    /// Read data from a host-visible buffer.
    pub(crate) fn read_from_buffer(
        &self,
        memory: vk::DeviceMemory,
        size: usize,
    ) -> Result<Vec<u8>, InterpolateError> {
        // SAFETY: memory is a valid VkDeviceMemory handle allocated with
        // HOST_VISIBLE property. The mapped range [0, size) is within the
        // allocation. copy_nonoverlapping is sound because the mapped
        // pointer and the Vec's allocation are both valid for `size` bytes
        // and do not overlap. unmap_memory is called before returning.
        unsafe {
            let ptr = self
                .device
                .map_memory(memory, 0, size as u64, vk::MemoryMapFlags::empty())
                .map_err(|e| InterpolateError::InterpolateFailed(format!("map memory: {e}")))?;
            let mut data = vec![0u8; size];
            std::ptr::copy_nonoverlapping(ptr as *const u8, data.as_mut_ptr(), size);
            self.device.unmap_memory(memory);
            Ok(data)
        }
    }

    /// Submit a command buffer and wait for completion.
    pub(crate) fn submit_and_wait(&self, cmd: vk::CommandBuffer) -> Result<(), InterpolateError> {
        let fence_info = vk::FenceCreateInfo::default();
        // SAFETY: self.device is a valid handle and fence_info is
        // correctly initialized with default (unsignaled) state.
        let fence = unsafe { self.device.create_fence(&fence_info, None) }
            .map_err(|e| InterpolateError::InterpolateFailed(format!("create fence: {e}")))?;

        let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));

        // SAFETY: All Vulkan handles (device, queue, fence, command buffer)
        // are valid. The command buffer was allocated from this device's
        // command pool and has been ended. wait_for_fences with timeout
        // u64::MAX ensures the GPU has finished before we destroy the fence.
        unsafe {
            self.device
                .queue_submit(self.compute_queue, &[submit_info], fence)
                .map_err(|e| InterpolateError::InterpolateFailed(format!("queue submit: {e}")))?;
            self.device
                .wait_for_fences(&[fence], true, u64::MAX)
                .map_err(|e| InterpolateError::InterpolateFailed(format!("wait fence: {e}")))?;
            self.device.destroy_fence(fence, None);
        }

        Ok(())
    }

    /// Allocate a single command buffer from the pool.
    pub(crate) fn allocate_command_buffer(&self) -> Result<vk::CommandBuffer, InterpolateError> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        // SAFETY: self.device and self.command_pool are valid handles.
        // alloc_info specifies one primary command buffer from the pool.
        let cmd = unsafe { self.device.allocate_command_buffers(&alloc_info) }
            .map_err(|e| InterpolateError::InterpolateFailed(format!("alloc cmd: {e}")))?;
        Ok(cmd[0])
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        // SAFETY: All Vulkan handles are valid. Resources are destroyed in
        // reverse creation order: command pool, then device, then instance.
        // No GPU operations are in flight at drop time because the command
        // buffer is submitted synchronously with a fence wait.
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
    #[ignore] // requires Vulkan runtime
    fn vulkan_context_creation() {
        let ctx = VulkanContext::new(&[]);
        assert!(ctx.is_ok(), "failed: {:?}", ctx.err());
        let ctx = ctx.unwrap();
        assert!(!ctx.device_name.is_empty());
        assert!(ctx.vendor_id > 0);
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

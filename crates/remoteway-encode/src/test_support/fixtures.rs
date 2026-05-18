//! Synthetic frame generators in NV12 (Y plane + interleaved UV) format.
//!
//! All fixtures are CPU-side `Vec<u8>` — the GPU upload happens in the
//! per-backend test glue (`Fixture::as_input_frame` builds the
//! `VkImage`-backed `InputFrame` lazily, owned by the encoder run).
//!
//! NV12 layout: `width * height` bytes of Y (luma) plane, followed by
//! `width * height / 2` bytes of UV (chroma) plane where each 2-byte pair
//! covers a 2×2 block of pixels.

use std::sync::Arc;

use ash::vk;
use ash::vk::TaggedStructure;
use remoteway_vulkan::{VideoCodec, VulkanContext};

use crate::encoder::InputFrame;
use crate::test_support::validator::DecodedFrame;

/// A pre-generated YUV frame ready for upload + encode.
pub struct Fixture {
    pub width: u32,
    pub height: u32,
    pub y: Vec<u8>,
    pub uv: Vec<u8>,
    /// Owned VkImage + memory once `upload_to_gpu` has been called.
    gpu: Option<GpuImage>,
}

struct GpuImage {
    ctx: Arc<VulkanContext>,
    image: vk::Image,
    memory: vk::DeviceMemory,
    _view: vk::ImageView,
}

impl Drop for GpuImage {
    fn drop(&mut self) {
        unsafe {
            let _ = self.ctx.device.device_wait_idle();
            self.ctx.device.destroy_image_view(self._view, None);
            self.ctx.device.destroy_image(self.image, None);
            self.ctx.device.free_memory(self.memory, None);
        }
    }
}

impl Fixture {
    pub fn gradient(width: u32, height: u32) -> Self {
        let mut me = Self::blank(width, height);
        me.y = gradient_luma(width, height);
        me.uv = neutral_chroma(width, height);
        me
    }

    /// Solid mid-gray fixture — diagnostic for encoder input correctness.
    pub fn solid_gray(width: u32, height: u32) -> Self {
        let mut me = Self::blank(width, height);
        me.y = vec![128u8; (width * height) as usize];
        me.uv = neutral_chroma(width, height);
        me
    }

    pub fn checkerboard(width: u32, height: u32) -> Self {
        let mut me = Self::blank(width, height);
        me.y = checkerboard_luma(width, height);
        me.uv = neutral_chroma(width, height);
        me
    }

    pub fn text_like(width: u32, height: u32) -> Self {
        let mut me = Self::blank(width, height);
        me.y = text_like_luma(width, height);
        me.uv = neutral_chroma(width, height);
        me
    }

    fn blank(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            y: vec![0; (width * height) as usize],
            uv: vec![128; (width * height / 2) as usize],
            gpu: None,
        }
    }

    /// View this fixture as if it had been decoded — for PSNR comparison.
    pub fn decoded_view(&self) -> DecodedFrame {
        DecodedFrame {
            width: self.width,
            height: self.height,
            y: self.y.clone(),
            u: chroma_split_u(&self.uv),
            v: chroma_split_v(&self.uv),
        }
    }

    /// Returns an `InputFrame` that points at a GPU-resident copy of this
    /// fixture's pixel data. The GPU image is allocated and uploaded on the
    /// first call; subsequent calls return the same image with only the PTS
    /// updated, which matches the typical test loop ("encode the same fixture
    /// as a sequence of frames").
    ///
    /// The image is created with NV12 (`G8_B8R8_2PLANE_420_UNORM`),
    /// `VIDEO_ENCODE_SRC` usage, exclusive sharing on the encode queue
    /// family, and the VideoProfileListInfoKHR for H.265 chained in (required
    /// by spec for video-encode-usage images).
    pub fn upload(&mut self, ctx: Arc<VulkanContext>, codec: VideoCodec) -> Result<(), String> {
        if self.gpu.is_some() {
            return Ok(());
        }
        let gpu = upload_nv12_image(&ctx, codec, self.width, self.height, &self.y, &self.uv)?;
        self.gpu = Some(GpuImage {
            ctx: ctx.clone(),
            image: gpu.image,
            memory: gpu.memory,
            _view: gpu.view,
        });
        Ok(())
    }

    pub fn as_input_frame(&self, pts: u64) -> InputFrame {
        let g = self
            .gpu
            .as_ref()
            .expect("Fixture::upload must be called before as_input_frame");
        InputFrame {
            image: g.image,
            width: self.width,
            height: self.height,
            pts_90khz: pts,
        }
    }
}

struct UploadedImage {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
}

fn upload_nv12_image(
    ctx: &VulkanContext,
    codec: VideoCodec,
    width: u32,
    height: u32,
    y: &[u8],
    uv: &[u8],
) -> Result<UploadedImage, String> {
    let encode_queue_family = ctx
        .video_encode_queue_family
        .ok_or_else(|| "no encode queue family".to_string())?;
    let _ = ctx
        .video_encode_queue
        .ok_or_else(|| "no encode queue".to_string())?;
    let compute_queue_family = ctx.queue_family;
    let compute_queue = ctx.compute_queue;

    // VideoProfileListInfoKHR is required for video-encode-usage images.
    let mut h265_profile = vk::VideoEncodeH265ProfileInfoKHR::default().std_profile_idc(
        ash::vk::native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN,
    );
    let mut profile = vk::VideoProfileInfoKHR::default()
        .video_codec_operation(codec.codec_operation())
        .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
        .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
        .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
        .push(&mut h265_profile);
    let profile_ref = &profile;
    let mut profile_list =
        vk::VideoProfileListInfoKHR::default().profiles(std::slice::from_ref(profile_ref));

    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::G8_B8R8_2PLANE_420_UNORM)
        .extent(vk::Extent3D { width, height, depth: 1 })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::VIDEO_ENCODE_SRC_KHR | vk::ImageUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .queue_family_indices(std::slice::from_ref(&encode_queue_family))
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .push(&mut profile_list);

    let image = unsafe { ctx.device.create_image(&image_info, None) }
        .map_err(|e| format!("create_image: {e:?}"))?;

    let req = unsafe { ctx.device.get_image_memory_requirements(image) };
    let mem_props = unsafe {
        ctx.instance
            .get_physical_device_memory_properties(ctx.physical_device)
    };
    let mem_type = (0..mem_props.memory_type_count)
        .find(|&i| {
            req.memory_type_bits & (1 << i) != 0
                && mem_props.memory_types[i as usize]
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        })
        .ok_or_else(|| "no device-local memory for nv12 image".to_string())?;
    let memory = unsafe {
        ctx.device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(mem_type),
            None,
        )
    }
    .map_err(|e| format!("allocate_memory: {e:?}"))?;
    unsafe {
        ctx.device
            .bind_image_memory(image, memory, 0)
            .map_err(|e| format!("bind_image_memory: {e:?}"))?;
    }

    let y_size = y.len() as u64;
    let uv_size = uv.len() as u64;
    let staging_size = y_size + uv_size;
    let (staging_buf, staging_mem) = ctx
        .create_host_buffer(staging_size, vk::BufferUsageFlags::TRANSFER_SRC)
        .map_err(|e| format!("staging buffer: {e:?}"))?;
    {
        let mut blob = Vec::with_capacity(staging_size as usize);
        blob.extend_from_slice(y);
        blob.extend_from_slice(uv);
        ctx.upload_to_buffer(staging_mem, &blob)
            .map_err(|e| format!("upload_to_buffer: {e:?}"))?;
    }

    // Upload via the compute/transfer queue family. The encode queue family
    // on AMD does NOT support TRANSFER ops; pipeline barriers and buffer→
    // image copies must run on a compute-capable queue. The image is created
    // with CONCURRENT sharing across compute + encode so no ownership
    // transfer is required.
    let xfer_family = compute_queue_family;
    let xfer_queue = compute_queue;
    let pool = unsafe {
        ctx.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(xfer_family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None,
        )
    }
    .map_err(|e| format!("create_command_pool: {e:?}"))?;
    let cmd = unsafe {
        ctx.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )
    }
    .map_err(|e| format!("allocate_command_buffers: {e:?}"))?[0];

    unsafe {
        ctx.device
            .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
            .map_err(|e| format!("begin_command_buffer: {e:?}"))?;

        // Barrier UNDEFINED → TRANSFER_DST_OPTIMAL (both planes).
        let to_xfer = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(
                        vk::ImageAspectFlags::PLANE_0 | vk::ImageAspectFlags::PLANE_1,
                    )
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
        ctx.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            std::slice::from_ref(&to_xfer),
        );

        // Copy plane 0 (Y).
        let y_copy = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::PLANE_0)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D { width, height, depth: 1 });
        ctx.device.cmd_copy_buffer_to_image(
            cmd,
            staging_buf,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            std::slice::from_ref(&y_copy),
        );

        // Copy plane 1 (UV at half resolution).
        let uv_copy = vk::BufferImageCopy::default()
            .buffer_offset(y_size)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::PLANE_1)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width: width / 2,
                height: height / 2,
                depth: 1,
            });
        ctx.device.cmd_copy_buffer_to_image(
            cmd,
            staging_buf,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            std::slice::from_ref(&uv_copy),
        );

        // Release: transfer-dst → encode-src with QFOT release from compute
        // to encode. Acquire half is issued by the encoder before reading.
        let release = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::VIDEO_ENCODE_SRC_KHR)
            .src_queue_family_index(compute_queue_family)
            .dst_queue_family_index(encode_queue_family)
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(
                        vk::ImageAspectFlags::PLANE_0 | vk::ImageAspectFlags::PLANE_1,
                    )
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::empty());
        ctx.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            std::slice::from_ref(&release),
        );

        ctx.device
            .end_command_buffer(cmd)
            .map_err(|e| format!("end_command_buffer: {e:?}"))?;

        let fence = ctx
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .map_err(|e| format!("create_fence: {e:?}"))?;
        let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
        ctx.device
            .queue_submit(xfer_queue, &[submit], fence)
            .map_err(|e| format!("queue_submit: {e:?}"))?;
        ctx.device
            .wait_for_fences(&[fence], true, u64::MAX)
            .map_err(|e| format!("wait_for_fences: {e:?}"))?;
        ctx.device.destroy_fence(fence, None);
        ctx.device.free_command_buffers(pool, &[cmd]);
        ctx.device.destroy_command_pool(pool, None);
        ctx.device.destroy_buffer(staging_buf, None);
        ctx.device.free_memory(staging_mem, None);

        // Acquire half of the QFOT on the encode queue. Without this the
        // image data uploaded on the compute family is not visible to the
        // encoder.
        let encode_queue = ctx.video_encode_queue.unwrap();
        let encode_pool = ctx
            .device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(encode_queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
            .map_err(|e| format!("acquire pool: {e:?}"))?;
        let acquire_cmd = ctx
            .device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(encode_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .map_err(|e| format!("acquire alloc: {e:?}"))?[0];
        ctx.device
            .begin_command_buffer(acquire_cmd, &vk::CommandBufferBeginInfo::default())
            .map_err(|e| format!("acquire begin: {e:?}"))?;
        let acquire = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::VIDEO_ENCODE_SRC_KHR)
            .src_queue_family_index(compute_queue_family)
            .dst_queue_family_index(encode_queue_family)
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(
                        vk::ImageAspectFlags::PLANE_0 | vk::ImageAspectFlags::PLANE_1,
                    )
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::MEMORY_READ);
        ctx.device.cmd_pipeline_barrier(
            acquire_cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            std::slice::from_ref(&acquire),
        );
        ctx.device
            .end_command_buffer(acquire_cmd)
            .map_err(|e| format!("acquire end: {e:?}"))?;
        let a_fence = ctx
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .map_err(|e| format!("acquire fence: {e:?}"))?;
        let a_submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&acquire_cmd));
        ctx.device
            .queue_submit(encode_queue, &[a_submit], a_fence)
            .map_err(|e| format!("acquire submit: {e:?}"))?;
        ctx.device
            .wait_for_fences(&[a_fence], true, u64::MAX)
            .map_err(|e| format!("acquire wait: {e:?}"))?;
        ctx.device.destroy_fence(a_fence, None);
        ctx.device.free_command_buffers(encode_pool, &[acquire_cmd]);
        ctx.device.destroy_command_pool(encode_pool, None);
    }

    let view = unsafe {
        ctx.device.create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::G8_B8R8_2PLANE_420_UNORM)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                ),
            None,
        )
    }
    .map_err(|e| format!("create_image_view: {e:?}"))?;

    Ok(UploadedImage {
        image,
        memory,
        view,
    })
}

/// Standalone helper that returns just the raw NV12 buffer for a gradient.
pub fn gradient_nv12(width: u32, height: u32) -> Vec<u8> {
    let mut buf = gradient_luma(width, height);
    buf.extend_from_slice(&neutral_chroma(width, height));
    buf
}

/// Standalone helper for a checkerboard NV12 buffer.
pub fn checkerboard_nv12(width: u32, height: u32) -> Vec<u8> {
    let mut buf = checkerboard_luma(width, height);
    buf.extend_from_slice(&neutral_chroma(width, height));
    buf
}

/// Standalone helper for a text-mimicking NV12 buffer.
pub fn text_like_nv12(width: u32, height: u32) -> Vec<u8> {
    let mut buf = text_like_luma(width, height);
    buf.extend_from_slice(&neutral_chroma(width, height));
    buf
}

fn gradient_luma(width: u32, height: u32) -> Vec<u8> {
    let mut y = Vec::with_capacity((width * height) as usize);
    for row in 0..height {
        for col in 0..width {
            // Smooth diagonal gradient — easy content, encoder should compress well.
            let val = ((row + col) * 255 / (width + height)) as u8;
            y.push(val);
        }
    }
    y
}

fn checkerboard_luma(width: u32, height: u32) -> Vec<u8> {
    let mut y = Vec::with_capacity((width * height) as usize);
    for row in 0..height {
        for col in 0..width {
            let cell = ((row / 16) + (col / 16)) % 2;
            y.push(if cell == 0 { 32 } else { 224 });
        }
    }
    y
}

fn text_like_luma(width: u32, height: u32) -> Vec<u8> {
    // Approximates UI text on a light background: 8-pixel-tall horizontal
    // bars of dark pixels interleaved with lighter rows. Hits the same
    // high-frequency content patterns that destroy quality in spatial
    // downscale, useful for PSNR sanity at moderate QP.
    let mut y = Vec::with_capacity((width * height) as usize);
    for row in 0..height {
        let is_text_row = (row / 8) % 3 == 0;
        for col in 0..width {
            let val = if is_text_row && (col / 4) % 5 != 0 {
                32 // "ink"
            } else {
                240 // "paper"
            };
            y.push(val);
        }
    }
    y
}

fn neutral_chroma(width: u32, height: u32) -> Vec<u8> {
    // 128 = neutral chroma (no color shift). Length is width*height/2 in NV12.
    vec![128; (width * height / 2) as usize]
}

fn chroma_split_u(uv: &[u8]) -> Vec<u8> {
    uv.iter().step_by(2).copied().collect()
}

fn chroma_split_v(uv: &[u8]) -> Vec<u8> {
    uv.iter().skip(1).step_by(2).copied().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_has_correct_size() {
        let buf = gradient_nv12(128, 96);
        // Y plane + UV plane = w*h*1.5
        assert_eq!(buf.len(), (128 * 96 * 3 / 2) as usize);
    }

    #[test]
    fn gradient_is_monotone_diagonally() {
        let f = Fixture::gradient(64, 64);
        // First pixel is darker than last
        assert!(f.y[0] < *f.y.last().unwrap());
    }

    #[test]
    fn checkerboard_has_two_distinct_values() {
        let f = Fixture::checkerboard(128, 128);
        let dark_count = f.y.iter().filter(|&&v| v == 32).count();
        let light_count = f.y.iter().filter(|&&v| v == 224).count();
        assert_eq!(dark_count + light_count, f.y.len());
        assert!(dark_count > 0 && light_count > 0);
    }

    #[test]
    fn text_like_has_high_contrast() {
        let f = Fixture::text_like(256, 64);
        assert!(f.y.contains(&32));
        assert!(f.y.contains(&240));
    }
}

//! H.264 (AVC) encoder backend using Vulkan Video encode KHR extensions.
//!
//! Pipeline: input NV12 `VkImage` → `VkVideoSessionKHR` → DPB pool → output
//! buffer → Annex-B bitstream readback. CQP rate control only; intra-refresh
//! not yet wired. Backed by `VK_KHR_video_encode_h264`.
//!
//! Layout mirrors the H.265 backend (`backends/h265.rs`) almost line-for-line.
//! The codec-specific differences are:
//! - No VPS — H.264 has only SPS + PPS in session parameters.
//! - `StdVideoEncodeH264*` structs replace the H.265 equivalents (different
//!   layouts: no temporal-id field on slice header, `frame_num` in addition to
//!   POC on picture info, etc.).
//! - `RefPicList0` is 32-byte array (vs 15 for H.265), padded with 0xff.

#![allow(clippy::too_many_lines)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::undocumented_unsafe_blocks)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_lossless)]
#![allow(non_upper_case_globals)]

use std::ffi::{c_char, CStr};
use std::sync::Arc;

use ash::vk;
use ash::vk::native as vkn;
use ash::vk::TaggedStructure;
use remoteway_vulkan::{VideoCodec, VulkanContext};

use crate::encoder::{EncodeParams, EncodedFrame, Encoder, FrameKind, InputFrame, RateControl};
use crate::EncodeError;

const SPS_ID: u8 = 0;
const PPS_ID: u8 = 0;
/// Two-slot DPB: one reference + one setup, the bare minimum for an
/// IDR-then-P-only low-latency loop.
const DPB_SLOTS: u32 = 2;
/// `VK_MAKE_VIDEO_STD_VERSION(1, 0, 0)`.
const STD_H264_ENCODE_API_VERSION_1_0_0: u32 = 1 << 22;
const STD_H264_ENCODE_EXTENSION_NAME: &CStr = c"VK_STD_vulkan_video_codec_h264_encode";

/// NV12 picture format consumed by the encoder.
const INPUT_FORMAT: vk::Format = vk::Format::G8_B8R8_2PLANE_420_UNORM;

/// Reasonable upper bound for one 720p frame's encoded payload (CQP 26).
const OUTPUT_BUFFER_BYTES: u64 = 1024 * 1024;

pub struct H264Encoder {
    ctx: Arc<VulkanContext>,
    params: EncodeParams,
    #[allow(dead_code)]
    caps_max_extent: vk::Extent2D,
    #[allow(dead_code)]
    encode_queue_family: u32,
    encode_queue: vk::Queue,

    video_queue: ash::khr::video_queue::Device,
    video_encode: ash::khr::video_encode_queue::Device,

    #[allow(dead_code)]
    profile: ProfileChain,

    session: vk::VideoSessionKHR,
    session_memory: Vec<vk::DeviceMemory>,
    session_params: vk::VideoSessionParametersKHR,

    /// SPS+PPS Annex-B bytes obtained from
    /// `vkGetEncodedVideoSessionParametersKHR`. Re-emitted in front of every
    /// keyframe in [`EncodedFrame::parameter_sets`].
    parameter_sets_blob: Vec<u8>,

    dpb: Vec<DpbSlot>,
    /// Index of the slot the previous P/IDR was written to. Next frame uses
    /// this as its reference; the new frame is written to the *other* slot.
    /// `None` until the first frame has been encoded.
    prior_ref_slot: Option<usize>,

    out_buffer: vk::Buffer,
    out_buffer_memory: vk::DeviceMemory,

    /// Two u32 result components per query: BUFFER_OFFSET + BYTES_WRITTEN.
    query_pool: vk::QueryPool,

    encode_cmd_pool: vk::CommandPool,
    encode_cmd: vk::CommandBuffer,
    encode_fence: vk::Fence,

    frame_index: u64,
    force_keyframe: bool,

    /// Cache of `VkImageView` per input `VkImage` to avoid re-creating (and
    /// leaking) a view on every `encode()`. The encoder destroys these in
    /// `Drop`. Images themselves are owned by the caller.
    input_views: std::collections::HashMap<vk::Image, vk::ImageView>,
}

struct DpbSlot {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
}

/// Owns the H.264 profile structs so their pointers stay valid for the lifetime
/// of the encoder.
struct ProfileChain {
    #[allow(dead_code)]
    h264: Box<vk::VideoEncodeH264ProfileInfoKHR<'static>>,
    profile: Box<vk::VideoProfileInfoKHR<'static>>,
    #[allow(dead_code)]
    profile_list: Box<vk::VideoProfileListInfoKHR<'static>>,
}

impl ProfileChain {
    fn new() -> Self {
        let h264: Box<vk::VideoEncodeH264ProfileInfoKHR<'static>> = Box::new(
            vk::VideoEncodeH264ProfileInfoKHR::default()
                .std_profile_idc(vkn::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH),
        );
        let mut profile: Box<vk::VideoProfileInfoKHR<'static>> = Box::new(
            vk::VideoProfileInfoKHR::default()
                .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H264)
                .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
                .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
                .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8),
        );
        profile.p_next = &*h264 as *const vk::VideoEncodeH264ProfileInfoKHR<'_>
            as *const std::ffi::c_void;

        let mut profile_list: Box<vk::VideoProfileListInfoKHR<'static>> =
            Box::new(vk::VideoProfileListInfoKHR::default());
        profile_list.profile_count = 1;
        profile_list.p_profiles = &*profile;

        Self {
            h264,
            profile,
            profile_list,
        }
    }

    fn profile(&self) -> &vk::VideoProfileInfoKHR<'static> {
        &self.profile
    }
}

impl Encoder for H264Encoder {
    fn new(ctx: Arc<VulkanContext>, params: EncodeParams) -> Result<Self, EncodeError> {
        if params.codec != VideoCodec::H264 {
            return Err(EncodeError::InvalidParams(format!(
                "H264Encoder cannot encode {:?}",
                params.codec
            )));
        }
        let caps = ctx.probe_video_encode_capabilities(VideoCodec::H264)?;
        params.validate_against(&caps)?;

        let encode_queue_family = ctx
            .video_encode_queue_family
            .ok_or_else(|| EncodeError::InvalidParams("context has no encode queue family".into()))?;
        let encode_queue = ctx.video_encode_queue.ok_or_else(|| {
            EncodeError::InvalidParams("context has no encode queue handle".into())
        })?;

        let video_queue = ash::khr::video_queue::Device::load(&ctx.instance, &ctx.device);
        let video_encode = ash::khr::video_encode_queue::Device::load(&ctx.instance, &ctx.device);

        let caps_max_extent = vk::Extent2D {
            width: caps.max_coded_extent.0,
            height: caps.max_coded_extent.1,
        };

        let profile = ProfileChain::new();

        // -- Vulkan Video session --
        let std_header = std_header_version();
        let coded_extent = vk::Extent2D {
            width: params.width,
            height: params.height,
        };
        let session_info = vk::VideoSessionCreateInfoKHR::default()
            .queue_family_index(encode_queue_family)
            .video_profile(profile.profile())
            .picture_format(INPUT_FORMAT)
            .max_coded_extent(coded_extent)
            .reference_picture_format(INPUT_FORMAT)
            .max_dpb_slots(DPB_SLOTS)
            .max_active_reference_pictures(1)
            .std_header_version(&std_header);
        let session = unsafe { video_queue.create_video_session(&session_info, None) }.map_err(
            |e| EncodeError::SubmitFailed(format!("create_video_session: {e:?}")),
        )?;

        // -- Bind session memory --
        let session_memory = bind_session_memory(&ctx, &video_queue, session)?;

        // -- Build SPS/PPS std structs --
        let sps_storage = build_sps(params.width, params.height);
        let pps_storage = build_pps();

        let h264_add = vk::VideoEncodeH264SessionParametersAddInfoKHR::default()
            .std_sp_ss(std::slice::from_ref(&sps_storage.sps))
            .std_pp_ss(std::slice::from_ref(&pps_storage.pps));

        let mut h264_params_info = vk::VideoEncodeH264SessionParametersCreateInfoKHR::default()
            .max_std_sps_count(1)
            .max_std_pps_count(1)
            .parameters_add_info(&h264_add);

        let session_params_info = vk::VideoSessionParametersCreateInfoKHR::default()
            .video_session(session)
            .push(&mut h264_params_info);

        let session_params = unsafe {
            video_queue.create_video_session_parameters(&session_params_info, None)
        }
        .map_err(|e| EncodeError::SubmitFailed(format!("create_video_session_parameters: {e:?}")))?;

        // -- Pull Annex-B blob for SPS/PPS --
        let parameter_sets_blob =
            fetch_parameter_sets_blob(&video_encode, session_params).map_err(|e| {
                EncodeError::ReadbackFailed(format!("get_encoded_video_session_parameters: {e:?}"))
            })?;
        // -- DPB pool --
        let dpb = create_dpb_pool(&ctx, &profile, coded_extent, encode_queue_family)?;

        // -- Output buffer --
        let (out_buffer, out_buffer_memory) =
            create_output_buffer(&ctx, &profile, OUTPUT_BUFFER_BYTES)?;

        // -- Query pool for encoded byte count --
        let query_pool = create_encode_query_pool(&ctx)?;

        // -- Encode command pool bound to encode queue family --
        let encode_cmd_pool = unsafe {
            ctx.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(encode_queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .map_err(|e| EncodeError::SubmitFailed(format!("create_command_pool: {e:?}")))?;

        let encode_cmd = unsafe {
            ctx.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(encode_cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(|e| EncodeError::SubmitFailed(format!("allocate_command_buffers: {e:?}")))?[0];

        let encode_fence = unsafe {
            ctx.device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        }
        .map_err(|e| EncodeError::SubmitFailed(format!("create_fence: {e:?}")))?;

        // -- Initialise DPB slots' layout to VIDEO_ENCODE_DPB --
        transition_dpb_to_initial_layout(
            &ctx,
            encode_cmd_pool,
            encode_queue,
            &dpb,
            encode_queue_family,
        )?;

        // -- One-shot RC init: Begin → control(RESET + ENCODE_RATE_CONTROL DISABLED) → End.
        // After this, the session's persistent RC state is DISABLED, so every
        // subsequent Begin can chain DISABLED and match.
        initialize_rate_control(
            &ctx,
            &video_queue,
            encode_cmd_pool,
            encode_queue,
            session,
            session_params,
        )?;

        Ok(Self {
            ctx,
            params,
            caps_max_extent,
            encode_queue_family,
            encode_queue,
            video_queue,
            video_encode,
            profile,
            session,
            session_memory,
            session_params,
            parameter_sets_blob,
            dpb,
            prior_ref_slot: None,
            out_buffer,
            out_buffer_memory,
            query_pool,
            encode_cmd_pool,
            encode_cmd,
            encode_fence,
            frame_index: 0,
            force_keyframe: false,
            input_views: std::collections::HashMap::new(),
        })
    }

    fn accepted_input_format(&self) -> vk::Format {
        INPUT_FORMAT
    }

    fn encode(&mut self, frame: InputFrame) -> Result<EncodedFrame, EncodeError> {
        if frame.width != self.params.width || frame.height != self.params.height {
            return Err(EncodeError::InvalidParams(format!(
                "InputFrame {}x{} != session {}x{}",
                frame.width, frame.height, self.params.width, self.params.height
            )));
        }

        let is_idr = self.frame_index == 0 || self.force_keyframe;
        self.force_keyframe = false;

        // Slot indices: if no prior ref we use slot 0; otherwise rotate.
        let setup_slot = match self.prior_ref_slot {
            None => 0,
            Some(prev) => (prev + 1) % self.dpb.len(),
        };
        let ref_slot = if is_idr { None } else { self.prior_ref_slot };

        let pic_order_cnt = (self.frame_index as i32) * 2; // POC type 0: step by 2
        // frame_num: 0 for IDR, otherwise increments modulo 2^(log2_max_frame_num_minus4 + 4).
        // We use log2_max_frame_num_minus4 = 0 → frame_num range is 0..16.
        let frame_num: u32 = if is_idr {
            0
        } else {
            (self.frame_index as u32) & 0xF
        };

        // === Build STD H.264 picture info / slice header / ref-list ===
        let primary_pic_type = if is_idr {
            vkn::StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_IDR
        } else {
            vkn::StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_P
        };

        let slice_type = if is_idr {
            vkn::StdVideoH264SliceType_STD_VIDEO_H264_SLICE_TYPE_I
        } else {
            vkn::StdVideoH264SliceType_STD_VIDEO_H264_SLICE_TYPE_P
        };

        let std_slice_header = vkn::StdVideoEncodeH264SliceHeader {
            flags: vkn::StdVideoEncodeH264SliceHeaderFlags {
                _bitfield_align_1: [],
                _bitfield_1: vkn::StdVideoEncodeH264SliceHeaderFlags::new_bitfield_1(
                    /* direct_spatial_mv_pred_flag    */ 0,
                    /* num_ref_idx_active_override_flag*/ 0,
                    /* reserved                       */ 0,
                ),
            },
            first_mb_in_slice: 0,
            slice_type,
            slice_alpha_c0_offset_div2: 0,
            slice_beta_offset_div2: 0,
            slice_qp_delta: 0,
            reserved1: 0,
            cabac_init_idc: vkn::StdVideoH264CabacInitIdc_STD_VIDEO_H264_CABAC_INIT_IDC_0,
            disable_deblocking_filter_idc:
                vkn::StdVideoH264DisableDeblockingFilterIdc_STD_VIDEO_H264_DISABLE_DEBLOCKING_FILTER_IDC_DISABLED,
            pWeightTable: std::ptr::null(),
        };

        let ref_lists = vkn::StdVideoEncodeH264ReferenceListsInfo {
            flags: vkn::StdVideoEncodeH264ReferenceListsInfoFlags {
                _bitfield_align_1: [],
                _bitfield_1:
                    vkn::StdVideoEncodeH264ReferenceListsInfoFlags::new_bitfield_1(0, 0, 0),
            },
            num_ref_idx_l0_active_minus1: 0,
            num_ref_idx_l1_active_minus1: 0,
            // 0xff is the spec-mandated "unused" sentinel for these lists.
            RefPicList0: {
                let mut a = [0xffu8; 32];
                if let Some(prev) = ref_slot {
                    // Must match slot_index of pReferenceSlots[0] in encode_info.
                    a[0] = prev as u8;
                }
                a
            },
            RefPicList1: [0xffu8; 32],
            refList0ModOpCount: 0,
            refList1ModOpCount: 0,
            refPicMarkingOpCount: 0,
            reserved1: [0u8; 7],
            pRefList0ModOperations: std::ptr::null(),
            pRefList1ModOperations: std::ptr::null(),
            pRefPicMarkingOperations: std::ptr::null(),
        };

        let pic_flags_bf = vkn::StdVideoEncodeH264PictureInfoFlags::new_bitfield_1(
            /* IdrPicFlag                          */ if is_idr { 1 } else { 0 },
            /* is_reference                        */ 1,
            /* no_output_of_prior_pics_flag        */ 0,
            /* long_term_reference_flag            */ 0,
            /* adaptive_ref_pic_marking_mode_flag  */ 0,
            /* reserved                            */ 0,
        );

        let std_pic_info = vkn::StdVideoEncodeH264PictureInfo {
            flags: vkn::StdVideoEncodeH264PictureInfoFlags {
                _bitfield_align_1: [],
                _bitfield_1: pic_flags_bf,
            },
            seq_parameter_set_id: SPS_ID,
            pic_parameter_set_id: PPS_ID,
            idr_pic_id: 0,
            primary_pic_type,
            frame_num,
            PicOrderCnt: pic_order_cnt,
            temporal_id: 0,
            reserved1: [0; 3],
            pRefLists: &ref_lists,
        };

        let nalu_slice = vk::VideoEncodeH264NaluSliceInfoKHR::default()
            .constant_qp(self.params_qp())
            .std_slice_header(&std_slice_header);

        let h264_picture_info = vk::VideoEncodeH264PictureInfoKHR::default()
            .nalu_slice_entries(std::slice::from_ref(&nalu_slice))
            .std_picture_info(&std_pic_info);

        // === Reference setup & reference slot infos for vkCmdBeginVideoCoding ===
        let setup_std_ref = vkn::StdVideoEncodeH264ReferenceInfo {
            flags: vkn::StdVideoEncodeH264ReferenceInfoFlags {
                _bitfield_align_1: [],
                _bitfield_1:
                    vkn::StdVideoEncodeH264ReferenceInfoFlags::new_bitfield_1(0, 0),
            },
            primary_pic_type,
            FrameNum: frame_num,
            PicOrderCnt: pic_order_cnt,
            long_term_pic_num: 0,
            long_term_frame_idx: 0,
            temporal_id: 0,
        };
        let setup_h264 = vk::VideoEncodeH264DpbSlotInfoKHR::default()
            .std_reference_info(&setup_std_ref);

        let setup_pic_resource = vk::VideoPictureResourceInfoKHR::default()
            .coded_offset(vk::Offset2D { x: 0, y: 0 })
            .coded_extent(vk::Extent2D {
                width: self.params.width,
                height: self.params.height,
            })
            .base_array_layer(0)
            .image_view_binding(self.dpb[setup_slot].view);

        let mut setup_h264_slot = setup_h264;
        let setup_ref_info = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(setup_slot as i32)
            .picture_resource(&setup_pic_resource)
            .push(&mut setup_h264_slot);

        // The reference slot (for P frames): same shape but with a slot_index
        // matching the previously-set-up DPB.
        let ref_std_ref;
        let ref_h264;
        let ref_pic_resource;
        let mut ref_h264_slot;
        let reference_slots_storage: Option<vk::VideoReferenceSlotInfoKHR<'_>>;

        if let Some(prev) = ref_slot {
            ref_std_ref = vkn::StdVideoEncodeH264ReferenceInfo {
                flags: vkn::StdVideoEncodeH264ReferenceInfoFlags {
                    _bitfield_align_1: [],
                    _bitfield_1:
                        vkn::StdVideoEncodeH264ReferenceInfoFlags::new_bitfield_1(0, 0),
                },
                primary_pic_type: vkn::StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_P,
                FrameNum: ((self.frame_index as u32).wrapping_sub(1)) & 0xF,
                PicOrderCnt: ((self.frame_index as i32) - 1) * 2,
                long_term_pic_num: 0,
                long_term_frame_idx: 0,
                temporal_id: 0,
            };
            ref_h264 = vk::VideoEncodeH264DpbSlotInfoKHR::default()
                .std_reference_info(&ref_std_ref);
            ref_h264_slot = ref_h264;
            ref_pic_resource = vk::VideoPictureResourceInfoKHR::default()
                .coded_offset(vk::Offset2D { x: 0, y: 0 })
                .coded_extent(vk::Extent2D {
                    width: self.params.width,
                    height: self.params.height,
                })
                .base_array_layer(0)
                .image_view_binding(self.dpb[prev].view);
            reference_slots_storage = Some(
                vk::VideoReferenceSlotInfoKHR::default()
                    .slot_index(prev as i32)
                    .picture_resource(&ref_pic_resource)
                    .push(&mut ref_h264_slot),
            );
        } else {
            ref_std_ref = unsafe { std::mem::zeroed() };
            ref_h264 = vk::VideoEncodeH264DpbSlotInfoKHR::default();
            ref_h264_slot = ref_h264;
            ref_pic_resource = vk::VideoPictureResourceInfoKHR::default();
            reference_slots_storage = None;
        }

        // The Begin call needs both the setup slot (with slot_index = -1) and
        // any active reference slots. Assemble the begin_slots array now.
        let mut begin_slots: Vec<vk::VideoReferenceSlotInfoKHR<'_>> = Vec::new();

        let begin_setup_pic_resource = vk::VideoPictureResourceInfoKHR::default()
            .coded_offset(vk::Offset2D { x: 0, y: 0 })
            .coded_extent(vk::Extent2D {
                width: self.params.width,
                height: self.params.height,
            })
            .base_array_layer(0)
            .image_view_binding(self.dpb[setup_slot].view);
        // VkVideoEncodeH264DpbSlotInfoKHR REQUIRES non-null pStdReferenceInfo
        // even for the "to-be-setup" slot in BeginCoding.
        let begin_setup_std_ref = vkn::StdVideoEncodeH264ReferenceInfo {
            flags: vkn::StdVideoEncodeH264ReferenceInfoFlags {
                _bitfield_align_1: [],
                _bitfield_1: vkn::StdVideoEncodeH264ReferenceInfoFlags::new_bitfield_1(0, 0),
            },
            primary_pic_type,
            FrameNum: frame_num,
            PicOrderCnt: pic_order_cnt,
            long_term_pic_num: 0,
            long_term_frame_idx: 0,
            temporal_id: 0,
        };
        let mut begin_setup_h264 = vk::VideoEncodeH264DpbSlotInfoKHR::default()
            .std_reference_info(&begin_setup_std_ref);
        // Setup slot in BeginCoding uses slot_index = -1 because the slot
        // becomes active only after the encode op writes to it.
        let setup_in_begin = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(-1)
            .picture_resource(&begin_setup_pic_resource)
            .push(&mut begin_setup_h264);
        begin_slots.push(setup_in_begin);

        if let Some(ref_info) = &reference_slots_storage {
            begin_slots.push(*ref_info);
        }

        // Begin scope MUST chain VkVideoEncodeRateControlInfoKHR with mode =
        // DISABLED to match the session's persistent RC state set via
        // cmd_control. Without this, the validator treats the scope as DEFAULT
        // mode, which then conflicts with per-slice constant_qp != 0.
        let mut begin_rc_h264 = vk::VideoEncodeH264RateControlInfoKHR::default()
            .flags(vk::VideoEncodeH264RateControlFlagsKHR::REGULAR_GOP)
            .gop_frame_count(u32::MAX)
            .idr_period(u32::MAX)
            .consecutive_b_frame_count(0)
            .temporal_layer_count(1);
        let mut begin_rc = vk::VideoEncodeRateControlInfoKHR::default()
            .rate_control_mode(vk::VideoEncodeRateControlModeFlagsKHR::DISABLED);
        let begin_info_with_rc = vk::VideoBeginCodingInfoKHR::default()
            .video_session(self.session)
            .video_session_parameters(self.session_params)
            .reference_slots(&begin_slots)
            .push(&mut begin_rc)
            .push(&mut begin_rc_h264);

        // === Record command buffer ===
        unsafe {
            self.ctx
                .device
                .reset_command_pool(self.encode_cmd_pool, vk::CommandPoolResetFlags::empty())
                .map_err(|e| EncodeError::SubmitFailed(format!("reset_command_pool: {e:?}")))?;
            self.ctx
                .device
                .begin_command_buffer(self.encode_cmd, &vk::CommandBufferBeginInfo::default())
                .map_err(|e| EncodeError::SubmitFailed(format!("begin_command_buffer: {e:?}")))?;
        }

        // Reset query pool entries (one query per frame).
        unsafe {
            self.ctx
                .device
                .cmd_reset_query_pool(self.encode_cmd, self.query_pool, 0, 1);
        }

        // Input image is expected to already be in VIDEO_ENCODE_SRC layout
        // (the caller uploads and transitions it).

        unsafe {
            self.video_queue
                .cmd_begin_video_coding(self.encode_cmd, &begin_info_with_rc);
        }

        // Begin query for byte feedback.
        unsafe {
            self.ctx.device.cmd_begin_query(
                self.encode_cmd,
                self.query_pool,
                0,
                vk::QueryControlFlags::empty(),
            );
        }

        let input_view = if let Some(v) = self.input_views.get(&frame.image) {
            *v
        } else {
            let v = create_image_view(&self.ctx, frame.image, INPUT_FORMAT, vk::ImageAspectFlags::COLOR)?;
            self.input_views.insert(frame.image, v);
            v
        };
        let src_pic_resource = vk::VideoPictureResourceInfoKHR::default()
            .coded_offset(vk::Offset2D { x: 0, y: 0 })
            .coded_extent(vk::Extent2D {
                width: self.params.width,
                height: self.params.height,
            })
            .base_array_layer(0)
            .image_view_binding(input_view);

        let mut h264_picture_info_mut = h264_picture_info;
        let mut encode_info = vk::VideoEncodeInfoKHR::default()
            .dst_buffer(self.out_buffer)
            .dst_buffer_offset(0)
            .dst_buffer_range(OUTPUT_BUFFER_BYTES)
            .src_picture_resource(src_pic_resource)
            .setup_reference_slot(&setup_ref_info)
            .push(&mut h264_picture_info_mut);

        let ref_for_encode_storage;
        if let Some(prev) = ref_slot {
            ref_for_encode_storage = vk::VideoReferenceSlotInfoKHR::default()
                .slot_index(prev as i32)
                .picture_resource(&ref_pic_resource)
                .push(&mut ref_h264_slot);
            encode_info = encode_info.reference_slots(std::slice::from_ref(&ref_for_encode_storage));
        }

        unsafe {
            self.video_encode
                .cmd_encode_video(self.encode_cmd, &encode_info);
            self.ctx
                .device
                .cmd_end_query(self.encode_cmd, self.query_pool, 0);
        }

        let end_info = vk::VideoEndCodingInfoKHR::default();
        unsafe {
            self.video_queue
                .cmd_end_video_coding(self.encode_cmd, &end_info);

            self.ctx
                .device
                .end_command_buffer(self.encode_cmd)
                .map_err(|e| EncodeError::SubmitFailed(format!("end_command_buffer: {e:?}")))?;
        }

        // === Submit and wait ===
        unsafe {
            self.ctx
                .device
                .reset_fences(std::slice::from_ref(&self.encode_fence))
                .map_err(|e| EncodeError::SubmitFailed(format!("reset_fences: {e:?}")))?;
            let submit = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&self.encode_cmd));
            self.ctx
                .device
                .queue_submit(self.encode_queue, &[submit], self.encode_fence)
                .map_err(|e| EncodeError::SubmitFailed(format!("queue_submit: {e:?}")))?;
            self.ctx
                .device
                .wait_for_fences(&[self.encode_fence], true, u64::MAX)
                .map_err(|e| EncodeError::SubmitFailed(format!("wait_for_fences: {e:?}")))?;
        }

        // === Read encoded byte count ===
        // Feedback components are returned in bit-position order: BUFFER_OFFSET
        // (bit 0) then BYTES_WRITTEN (bit 1). Package as [[u32; 2]; 1] so ash
        // uses queryCount=1, stride=8, not queryCount=2, stride=4.
        let mut feedback: [[u32; 2]; 1] = [[0u32; 2]; 1];
        unsafe {
            self.ctx
                .device
                .get_query_pool_results(
                    self.query_pool,
                    0,
                    &mut feedback,
                    vk::QueryResultFlags::WAIT,
                )
                .map_err(|e| EncodeError::ReadbackFailed(format!("get_query_pool_results: {e:?}")))?;
        }
        let _buffer_offset = feedback[0][0] as u64;
        let bytes_written = feedback[0][1] as u64;

        // === Read bitstream bytes ===
        let bitstream = {
            let read_size = bytes_written.min(OUTPUT_BUFFER_BYTES) as usize;
            let mut out = vec![0u8; read_size];
            unsafe {
                let ptr = self
                    .ctx
                    .device
                    .map_memory(
                        self.out_buffer_memory,
                        0,
                        read_size as u64,
                        vk::MemoryMapFlags::empty(),
                    )
                    .map_err(|e| {
                        EncodeError::ReadbackFailed(format!("map_memory(out_buffer): {e:?}"))
                    })?;
                std::ptr::copy_nonoverlapping(ptr as *const u8, out.as_mut_ptr(), read_size);
                self.ctx.device.unmap_memory(self.out_buffer_memory);
            }
            out
        };

        let parameter_sets = if is_idr {
            Some(self.parameter_sets_blob.clone())
        } else {
            None
        };

        let kind = if is_idr { FrameKind::Idr } else { FrameKind::P };

        self.frame_index += 1;
        self.prior_ref_slot = Some(setup_slot);

        Ok(EncodedFrame {
            data: bitstream,
            kind,
            pts_90khz: frame.pts_90khz,
            parameter_sets,
        })
    }

    fn request_keyframe(&mut self) {
        self.force_keyframe = true;
    }

    fn params(&self) -> &EncodeParams {
        &self.params
    }
}

impl H264Encoder {
    fn params_qp(&self) -> i32 {
        match self.params.rate_control {
            RateControl::ConstantQp { qp } => qp as i32,
            // Non-CQP modes are validated against device caps; if validation
            // passed, the encoder should be using DISABLED rate control mode
            // and supplying QP at slice level anyway. Default to 26.
            _ => 26,
        }
    }
}

impl Drop for H264Encoder {
    fn drop(&mut self) {
        unsafe {
            let _ = self.ctx.device.device_wait_idle();
            for (_, v) in self.input_views.drain() {
                self.ctx.device.destroy_image_view(v, None);
            }
            self.ctx.device.destroy_fence(self.encode_fence, None);
            self.ctx
                .device
                .destroy_command_pool(self.encode_cmd_pool, None);
            self.ctx.device.destroy_query_pool(self.query_pool, None);
            self.ctx.device.destroy_buffer(self.out_buffer, None);
            self.ctx
                .device
                .free_memory(self.out_buffer_memory, None);
            for slot in self.dpb.drain(..) {
                self.ctx.device.destroy_image_view(slot.view, None);
                self.ctx.device.destroy_image(slot.image, None);
                self.ctx.device.free_memory(slot.memory, None);
            }
            self.video_queue
                .destroy_video_session_parameters(self.session_params, None);
            self.video_queue.destroy_video_session(self.session, None);
            for mem in self.session_memory.drain(..) {
                self.ctx.device.free_memory(mem, None);
            }
        }
    }
}

// =============================================================================
// Helpers — std header / parameter sets
// =============================================================================

fn std_header_version() -> vk::ExtensionProperties {
    let mut props = vk::ExtensionProperties {
        extension_name: [0; vk::MAX_EXTENSION_NAME_SIZE],
        spec_version: STD_H264_ENCODE_API_VERSION_1_0_0,
    };
    let name = STD_H264_ENCODE_EXTENSION_NAME.to_bytes_with_nul();
    for (i, &b) in name.iter().enumerate() {
        props.extension_name[i] = b as c_char;
    }
    props
}

struct SpsStorage {
    sps: vkn::StdVideoH264SequenceParameterSet,
}

fn build_sps(width: u32, height: u32) -> SpsStorage {
    // log2_max_frame_num_minus4 = 0      → frame_num is 4 bits (0..16).
    // log2_max_pic_order_cnt_lsb_minus4 = 4 → POC LSB is 8 bits.
    let sps_flags = vkn::StdVideoH264SpsFlags {
        _bitfield_align_1: [],
        _bitfield_1: vkn::StdVideoH264SpsFlags::new_bitfield_1(
            /* constraint_set0_flag                  */ 0,
            /* constraint_set1_flag                  */ 0,
            /* constraint_set2_flag                  */ 0,
            /* constraint_set3_flag                  */ 0,
            /* constraint_set4_flag                  */ 0,
            /* constraint_set5_flag                  */ 0,
            /* direct_8x8_inference_flag             */ 1,
            /* mb_adaptive_frame_field_flag          */ 0,
            /* frame_mbs_only_flag                   */ 1,
            /* delta_pic_order_always_zero_flag      */ 0,
            /* separate_colour_plane_flag            */ 0,
            /* gaps_in_frame_num_value_allowed_flag  */ 0,
            /* qpprime_y_zero_transform_bypass_flag  */ 0,
            /* frame_cropping_flag                   */ 0,
            /* seq_scaling_matrix_present_flag       */ 0,
            /* vui_parameters_present_flag           */ 0,
        ),
        __bindgen_padding_0: 0,
    };

    let sps = vkn::StdVideoH264SequenceParameterSet {
        flags: sps_flags,
        profile_idc: vkn::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH,
        level_idc: vkn::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_4_1,
        chroma_format_idc: vkn::StdVideoH264ChromaFormatIdc_STD_VIDEO_H264_CHROMA_FORMAT_IDC_420,
        seq_parameter_set_id: SPS_ID,
        bit_depth_luma_minus8: 0,
        bit_depth_chroma_minus8: 0,
        log2_max_frame_num_minus4: 0,
        pic_order_cnt_type: vkn::StdVideoH264PocType_STD_VIDEO_H264_POC_TYPE_0,
        offset_for_non_ref_pic: 0,
        offset_for_top_to_bottom_field: 0,
        log2_max_pic_order_cnt_lsb_minus4: 4,
        num_ref_frames_in_pic_order_cnt_cycle: 0,
        max_num_ref_frames: 1,
        reserved1: 0,
        pic_width_in_mbs_minus1: (width / 16).saturating_sub(1),
        pic_height_in_map_units_minus1: (height / 16).saturating_sub(1),
        frame_crop_left_offset: 0,
        frame_crop_right_offset: 0,
        frame_crop_top_offset: 0,
        frame_crop_bottom_offset: 0,
        reserved2: 0,
        pOffsetForRefFrame: std::ptr::null(),
        pScalingLists: std::ptr::null(),
        pSequenceParameterSetVui: std::ptr::null(),
    };

    SpsStorage { sps }
}

struct PpsStorage {
    pps: vkn::StdVideoH264PictureParameterSet,
}

fn build_pps() -> PpsStorage {
    let pps_flags = vkn::StdVideoH264PpsFlags {
        _bitfield_align_1: [],
        _bitfield_1: vkn::StdVideoH264PpsFlags::new_bitfield_1(
            /* transform_8x8_mode_flag                          */ 1,
            /* redundant_pic_cnt_present_flag                   */ 0,
            /* constrained_intra_pred_flag                      */ 0,
            /* deblocking_filter_control_present_flag           */ 1,
            /* weighted_pred_flag                               */ 0,
            /* bottom_field_pic_order_in_frame_present_flag     */ 0,
            /* entropy_coding_mode_flag (CABAC)                 */ 1,
            /* pic_scaling_matrix_present_flag                  */ 0,
        ),
        __bindgen_padding_0: [0; 3],
    };

    let pps = vkn::StdVideoH264PictureParameterSet {
        flags: pps_flags,
        seq_parameter_set_id: SPS_ID,
        pic_parameter_set_id: PPS_ID,
        num_ref_idx_l0_default_active_minus1: 0,
        num_ref_idx_l1_default_active_minus1: 0,
        weighted_bipred_idc:
            vkn::StdVideoH264WeightedBipredIdc_STD_VIDEO_H264_WEIGHTED_BIPRED_IDC_DEFAULT,
        pic_init_qp_minus26: 0,
        pic_init_qs_minus26: 0,
        chroma_qp_index_offset: 0,
        second_chroma_qp_index_offset: 0,
        pScalingLists: std::ptr::null(),
    };
    PpsStorage { pps }
}

// =============================================================================
// Helpers — Vulkan object creation
// =============================================================================

fn bind_session_memory(
    ctx: &VulkanContext,
    video_queue: &ash::khr::video_queue::Device,
    session: vk::VideoSessionKHR,
) -> Result<Vec<vk::DeviceMemory>, EncodeError> {
    let count = unsafe { video_queue.get_video_session_memory_requirements_len(session) }
        .map_err(|e| EncodeError::SubmitFailed(format!("session_memory_len: {e:?}")))?;
    let mut reqs = vec![vk::VideoSessionMemoryRequirementsKHR::default(); count];
    unsafe { video_queue.get_video_session_memory_requirements(session, &mut reqs) }
        .map_err(|e| EncodeError::SubmitFailed(format!("session_memory_reqs: {e:?}")))?;

    let mem_props = unsafe {
        ctx.instance
            .get_physical_device_memory_properties(ctx.physical_device)
    };

    let mut allocations = Vec::with_capacity(reqs.len());
    let mut binds = Vec::with_capacity(reqs.len());

    for req in &reqs {
        let mem_type = (0..mem_props.memory_type_count)
            .find(|&i| {
                req.memory_requirements.memory_type_bits & (1 << i) != 0
                    && mem_props.memory_types[i as usize]
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            })
            .ok_or_else(|| EncodeError::SubmitFailed("no device-local mem for session".into()))?;
        let mem = unsafe {
            ctx.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.memory_requirements.size)
                    .memory_type_index(mem_type),
                None,
            )
        }
        .map_err(|e| EncodeError::SubmitFailed(format!("session allocate_memory: {e:?}")))?;
        allocations.push(mem);
        binds.push(
            vk::BindVideoSessionMemoryInfoKHR::default()
                .memory_bind_index(req.memory_bind_index)
                .memory(mem)
                .memory_offset(0)
                .memory_size(req.memory_requirements.size),
        );
    }
    unsafe {
        video_queue
            .bind_video_session_memory(session, &binds)
            .map_err(|e| {
                EncodeError::SubmitFailed(format!("bind_video_session_memory: {e:?}"))
            })?;
    }

    Ok(allocations)
}

fn fetch_parameter_sets_blob(
    video_encode: &ash::khr::video_encode_queue::Device,
    session_params: vk::VideoSessionParametersKHR,
) -> Result<Vec<u8>, vk::Result> {
    let mut h264_get = vk::VideoEncodeH264SessionParametersGetInfoKHR::default()
        .write_std_sps(true)
        .write_std_pps(true)
        .std_sps_id(SPS_ID as u32)
        .std_pps_id(PPS_ID as u32);
    let get_info = vk::VideoEncodeSessionParametersGetInfoKHR::default()
        .video_session_parameters(session_params)
        .push(&mut h264_get);

    let mut feedback_h264 = vk::VideoEncodeH264SessionParametersFeedbackInfoKHR::default();
    let mut feedback = vk::VideoEncodeSessionParametersFeedbackInfoKHR::default()
        .push(&mut feedback_h264);

    let len = unsafe {
        video_encode.get_encoded_video_session_parameters_len(&get_info, Some(&mut feedback))?
    };
    let mut out: Vec<std::mem::MaybeUninit<u8>> = Vec::with_capacity(len);
    out.resize(len, std::mem::MaybeUninit::uninit());
    unsafe {
        video_encode.get_encoded_video_session_parameters(
            &get_info,
            Some(&mut feedback),
            &mut out,
        )?;
    }
    let init: Vec<u8> = out
        .into_iter()
        .map(|m| unsafe { m.assume_init() })
        .collect();
    Ok(init)
}

fn create_dpb_pool(
    ctx: &VulkanContext,
    profile: &ProfileChain,
    extent: vk::Extent2D,
    queue_family: u32,
) -> Result<Vec<DpbSlot>, EncodeError> {
    let mut out = Vec::with_capacity(DPB_SLOTS as usize);
    for _ in 0..DPB_SLOTS {
        let mut profile_list_holder: vk::VideoProfileListInfoKHR<'_> =
            unsafe { std::mem::zeroed() };
        profile_list_holder.s_type = vk::VideoProfileListInfoKHR::STRUCTURE_TYPE;
        profile_list_holder.profile_count = 1;
        profile_list_holder.p_profiles = profile.profile();
        let mut info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(INPUT_FORMAT)
            .extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::VIDEO_ENCODE_DPB_KHR)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(std::slice::from_ref(&queue_family))
            .initial_layout(vk::ImageLayout::UNDEFINED);
        info = info.push(&mut profile_list_holder);

        let image = unsafe { ctx.device.create_image(&info, None) }
            .map_err(|e| EncodeError::SubmitFailed(format!("dpb create_image: {e:?}")))?;

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
            .ok_or_else(|| EncodeError::SubmitFailed("dpb no device-local memory".into()))?;
        let memory = unsafe {
            ctx.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(mem_type),
                None,
            )
        }
        .map_err(|e| EncodeError::SubmitFailed(format!("dpb allocate_memory: {e:?}")))?;
        unsafe {
            ctx.device
                .bind_image_memory(image, memory, 0)
                .map_err(|e| EncodeError::SubmitFailed(format!("dpb bind_image: {e:?}")))?;
        }
        let view = create_image_view(ctx, image, INPUT_FORMAT, vk::ImageAspectFlags::COLOR)?;
        out.push(DpbSlot {
            image,
            memory,
            view,
        });
    }
    Ok(out)
}

fn create_image_view(
    ctx: &VulkanContext,
    image: vk::Image,
    format: vk::Format,
    aspect: vk::ImageAspectFlags,
) -> Result<vk::ImageView, EncodeError> {
    let info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(aspect)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        );
    unsafe { ctx.device.create_image_view(&info, None) }
        .map_err(|e| EncodeError::SubmitFailed(format!("create_image_view: {e:?}")))
}

fn create_output_buffer(
    ctx: &VulkanContext,
    profile: &ProfileChain,
    size: u64,
) -> Result<(vk::Buffer, vk::DeviceMemory), EncodeError> {
    let mut profile_list_holder: vk::VideoProfileListInfoKHR<'_> = unsafe { std::mem::zeroed() };
    profile_list_holder.s_type = vk::VideoProfileListInfoKHR::STRUCTURE_TYPE;
    profile_list_holder.profile_count = 1;
    profile_list_holder.p_profiles = profile.profile();

    let mut info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(vk::BufferUsageFlags::VIDEO_ENCODE_DST_KHR)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    info = info.push(&mut profile_list_holder);

    let buffer = unsafe { ctx.device.create_buffer(&info, None) }
        .map_err(|e| EncodeError::SubmitFailed(format!("out create_buffer: {e:?}")))?;
    let req = unsafe { ctx.device.get_buffer_memory_requirements(buffer) };
    let mem_props = unsafe {
        ctx.instance
            .get_physical_device_memory_properties(ctx.physical_device)
    };
    let mem_type = (0..mem_props.memory_type_count)
        .find(|&i| {
            req.memory_type_bits & (1 << i) != 0
                && mem_props.memory_types[i as usize]
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
                && mem_props.memory_types[i as usize]
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
        })
        .ok_or_else(|| EncodeError::SubmitFailed("out buffer: no host-visible memory".into()))?;
    let memory = unsafe {
        ctx.device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(mem_type),
            None,
        )
    }
    .map_err(|e| EncodeError::SubmitFailed(format!("out allocate_memory: {e:?}")))?;
    unsafe {
        ctx.device
            .bind_buffer_memory(buffer, memory, 0)
            .map_err(|e| EncodeError::SubmitFailed(format!("out bind_buffer: {e:?}")))?;
    }
    Ok((buffer, memory))
}

fn create_encode_query_pool(ctx: &VulkanContext) -> Result<vk::QueryPool, EncodeError> {
    // Build the chain bottom-up so each `push` sees a null p_next.
    let mut h264_profile_for_query = vk::VideoEncodeH264ProfileInfoKHR::default()
        .std_profile_idc(vkn::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH);
    let mut profile_for_query = vk::VideoProfileInfoKHR::default()
        .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H264)
        .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
        .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
        .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8);
    let mut feedback_info = vk::QueryPoolVideoEncodeFeedbackCreateInfoKHR::default()
        .encode_feedback_flags(
            vk::VideoEncodeFeedbackFlagsKHR::BITSTREAM_BYTES_WRITTEN
                | vk::VideoEncodeFeedbackFlagsKHR::BITSTREAM_BUFFER_OFFSET,
        );
    let info = vk::QueryPoolCreateInfo::default()
        .query_type(vk::QueryType::VIDEO_ENCODE_FEEDBACK_KHR)
        .query_count(1)
        .push(&mut feedback_info)
        .push(&mut h264_profile_for_query)
        .push(&mut profile_for_query);

    unsafe { ctx.device.create_query_pool(&info, None) }
        .map_err(|e| EncodeError::SubmitFailed(format!("create_query_pool: {e:?}")))
}

fn initialize_rate_control(
    ctx: &VulkanContext,
    video_queue: &ash::khr::video_queue::Device,
    pool: vk::CommandPool,
    queue: vk::Queue,
    session: vk::VideoSessionKHR,
    session_params: vk::VideoSessionParametersKHR,
) -> Result<(), EncodeError> {
    let cmd = unsafe {
        ctx.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )
    }
    .map_err(|e| EncodeError::SubmitFailed(format!("rc init alloc cmd: {e:?}")))?[0];
    unsafe {
        ctx.device
            .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
            .map_err(|e| EncodeError::SubmitFailed(format!("rc init begin: {e:?}")))?;
    }

    let begin = vk::VideoBeginCodingInfoKHR::default()
        .video_session(session)
        .video_session_parameters(session_params);
    unsafe {
        video_queue.cmd_begin_video_coding(cmd, &begin);
    }

    let mut rc_h264 = vk::VideoEncodeH264RateControlInfoKHR::default()
        .flags(vk::VideoEncodeH264RateControlFlagsKHR::REGULAR_GOP)
        .gop_frame_count(u32::MAX)
        .idr_period(u32::MAX)
        .consecutive_b_frame_count(0)
        .temporal_layer_count(1);
    let mut rc = vk::VideoEncodeRateControlInfoKHR::default()
        .rate_control_mode(vk::VideoEncodeRateControlModeFlagsKHR::DISABLED);
    let control = vk::VideoCodingControlInfoKHR::default()
        .flags(
            vk::VideoCodingControlFlagsKHR::RESET
                | vk::VideoCodingControlFlagsKHR::ENCODE_RATE_CONTROL,
        )
        .push(&mut rc)
        .push(&mut rc_h264);
    unsafe {
        video_queue.cmd_control_video_coding(cmd, &control);
    }

    let end = vk::VideoEndCodingInfoKHR::default();
    unsafe {
        video_queue.cmd_end_video_coding(cmd, &end);
        ctx.device
            .end_command_buffer(cmd)
            .map_err(|e| EncodeError::SubmitFailed(format!("rc init end: {e:?}")))?;
        let fence = ctx
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .map_err(|e| EncodeError::SubmitFailed(format!("rc init fence: {e:?}")))?;
        let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
        ctx.device
            .queue_submit(queue, &[submit], fence)
            .map_err(|e| EncodeError::SubmitFailed(format!("rc init submit: {e:?}")))?;
        ctx.device
            .wait_for_fences(&[fence], true, u64::MAX)
            .map_err(|e| EncodeError::SubmitFailed(format!("rc init wait: {e:?}")))?;
        ctx.device.destroy_fence(fence, None);
        ctx.device.free_command_buffers(pool, &[cmd]);
    }
    Ok(())
}

fn transition_dpb_to_initial_layout(
    ctx: &VulkanContext,
    pool: vk::CommandPool,
    queue: vk::Queue,
    dpb: &[DpbSlot],
    _queue_family: u32,
) -> Result<(), EncodeError> {
    let cmd = unsafe {
        ctx.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )
    }
    .map_err(|e| EncodeError::SubmitFailed(format!("dpb init alloc cmd: {e:?}")))?[0];
    unsafe {
        ctx.device
            .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
            .map_err(|e| EncodeError::SubmitFailed(format!("dpb begin: {e:?}")))?;
    }
    for slot in dpb {
        cmd_transition(
            ctx,
            cmd,
            slot.image,
            vk::ImageLayout::VIDEO_ENCODE_DPB_KHR,
            vk::ImageAspectFlags::COLOR,
        );
    }
    unsafe {
        ctx.device
            .end_command_buffer(cmd)
            .map_err(|e| EncodeError::SubmitFailed(format!("dpb end: {e:?}")))?;
        let fence = ctx
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .map_err(|e| EncodeError::SubmitFailed(format!("dpb fence: {e:?}")))?;
        let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
        ctx.device
            .queue_submit(queue, &[submit], fence)
            .map_err(|e| EncodeError::SubmitFailed(format!("dpb submit: {e:?}")))?;
        ctx.device
            .wait_for_fences(&[fence], true, u64::MAX)
            .map_err(|e| EncodeError::SubmitFailed(format!("dpb wait: {e:?}")))?;
        ctx.device.destroy_fence(fence, None);
        ctx.device.free_command_buffers(pool, &[cmd]);
    }
    Ok(())
}

fn cmd_transition(
    ctx: &VulkanContext,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    new_layout: vk::ImageLayout,
    aspect: vk::ImageAspectFlags,
) {
    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(aspect)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        )
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::empty());
    unsafe {
        ctx.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            std::slice::from_ref(&barrier),
        );
    }
}

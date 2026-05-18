//! H.265 (HEVC) encoder backend using Vulkan Video encode KHR extensions.
//!
//! Pipeline: input NV12 `VkImage` → `VkVideoSessionKHR` → DPB pool → output
//! buffer → Annex-B bitstream readback. CQP rate control only; intra-refresh
//! not yet wired. Backed by `VK_KHR_video_encode_h265`.
//!
//! Layout follows the standard Vulkan Video encode flow:
//! 1. `VkVideoSessionKHR` + memory binding
//! 2. `VkVideoSessionParametersKHR` carrying VPS/SPS/PPS (one each)
//! 3. DPB image pool with two reference slots (setup + ref) for low-latency P
//! 4. Reusable output buffer + query pool for encoded-byte readback
//! 5. Per-frame: layout transitions → `vkCmdBeginVideoCodingKHR` →
//!    `vkCmdEncodeVideoKHR` → `vkCmdEndVideoCodingKHR` → fence wait → readback

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

const VPS_ID: u8 = 0;
const SPS_ID: u8 = 0;
const PPS_ID: u8 = 0;
/// Two-slot DPB: one reference + one setup, the bare minimum for an
/// IDR-then-P-only low-latency loop.
const DPB_SLOTS: u32 = 2;
/// `VK_MAKE_VIDEO_STD_VERSION(1, 0, 0)`.
const STD_H265_ENCODE_API_VERSION_1_0_0: u32 = 1 << 22;
const STD_H265_ENCODE_EXTENSION_NAME: &CStr = c"VK_STD_vulkan_video_codec_h265_encode";

/// NV12 picture format consumed by the encoder.
const INPUT_FORMAT: vk::Format = vk::Format::G8_B8R8_2PLANE_420_UNORM;

/// Reasonable upper bound for one 720p frame's encoded payload (CQP 26).
/// Two passes of safety overhead; bitstream rarely exceeds half a megabyte.
const OUTPUT_BUFFER_BYTES: u64 = 1024 * 1024;

pub struct H265Encoder {
    ctx: Arc<VulkanContext>,
    params: EncodeParams,
    caps_max_extent: vk::Extent2D,
    encode_queue_family: u32,
    encode_queue: vk::Queue,

    video_queue: ash::khr::video_queue::Device,
    video_encode: ash::khr::video_encode_queue::Device,

    profile: ProfileChain,

    session: vk::VideoSessionKHR,
    session_memory: Vec<vk::DeviceMemory>,
    session_params: vk::VideoSessionParametersKHR,

    /// VPS+SPS+PPS Annex-B bytes obtained from
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

    /// Two queries per encode: BYTES_WRITTEN + STATUS (= 2 × u32).
    query_pool: vk::QueryPool,

    encode_cmd_pool: vk::CommandPool,
    encode_cmd: vk::CommandBuffer,
    encode_fence: vk::Fence,

    frame_index: u64,
    force_keyframe: bool,
    coding_initialized: bool,

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

/// Owns the H.265 profile structs so their pointers stay valid for the lifetime
/// of the encoder (the driver re-reads them on every `vkCmdBeginVideoCodingKHR`
/// via the video session's stored profile).
struct ProfileChain {
    h265: Box<vk::VideoEncodeH265ProfileInfoKHR<'static>>,
    profile: Box<vk::VideoProfileInfoKHR<'static>>,
    profile_list: Box<vk::VideoProfileListInfoKHR<'static>>,
}

impl ProfileChain {
    fn new() -> Self {
        let h265: Box<vk::VideoEncodeH265ProfileInfoKHR<'static>> = Box::new(
            vk::VideoEncodeH265ProfileInfoKHR::default()
                .std_profile_idc(vkn::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN),
        );
        let mut profile: Box<vk::VideoProfileInfoKHR<'static>> = Box::new(
            vk::VideoProfileInfoKHR::default()
                .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H265)
                .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
                .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
                .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8),
        );
        // Pin h265 in the chain. Box address is stable.
        profile.p_next = &*h265 as *const vk::VideoEncodeH265ProfileInfoKHR<'_>
            as *const std::ffi::c_void;

        let mut profile_list: Box<vk::VideoProfileListInfoKHR<'static>> =
            Box::new(vk::VideoProfileListInfoKHR::default());
        profile_list.profile_count = 1;
        profile_list.p_profiles = &*profile;

        Self {
            h265,
            profile,
            profile_list,
        }
    }

    fn profile(&self) -> &vk::VideoProfileInfoKHR<'static> {
        &self.profile
    }

    fn profile_list_ptr(&self) -> *const std::ffi::c_void {
        &*self.profile_list as *const _ as *const std::ffi::c_void
    }
}

impl Encoder for H265Encoder {
    fn new(ctx: Arc<VulkanContext>, params: EncodeParams) -> Result<Self, EncodeError> {
        if params.codec != VideoCodec::H265 {
            return Err(EncodeError::InvalidParams(format!(
                "H265Encoder cannot encode {:?}",
                params.codec
            )));
        }
        let caps = ctx.probe_video_encode_capabilities(VideoCodec::H265)?;
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

        // -- Build VPS/SPS/PPS std structs --
        let vps_storage = build_vps();
        let sps_storage = build_sps(params.width, params.height);
        let pps_storage = build_pps();

        let h265_add = vk::VideoEncodeH265SessionParametersAddInfoKHR::default()
            .std_vp_ss(std::slice::from_ref(&vps_storage.vps))
            .std_sp_ss(std::slice::from_ref(&sps_storage.sps))
            .std_pp_ss(std::slice::from_ref(&pps_storage.pps));

        let mut h265_params_info = vk::VideoEncodeH265SessionParametersCreateInfoKHR::default()
            .max_std_vps_count(1)
            .max_std_sps_count(1)
            .max_std_pps_count(1)
            .parameters_add_info(&h265_add);

        let session_params_info = vk::VideoSessionParametersCreateInfoKHR::default()
            .video_session(session)
            .push(&mut h265_params_info);

        let session_params = unsafe {
            video_queue.create_video_session_parameters(&session_params_info, None)
        }
        .map_err(|e| EncodeError::SubmitFailed(format!("create_video_session_parameters: {e:?}")))?;

        // -- Pull Annex-B blob for VPS/SPS/PPS --
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
        let query_pool = create_encode_query_pool(&ctx, &profile)?;

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
            coding_initialized: false,
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

        let pic_order_cnt = self.frame_index as i32;

        // === Build STD H.265 picture info / slice header / ref-list ===
        let pic_type = if is_idr {
            vkn::StdVideoH265PictureType_STD_VIDEO_H265_PICTURE_TYPE_IDR
        } else {
            vkn::StdVideoH265PictureType_STD_VIDEO_H265_PICTURE_TYPE_P
        };

        let slice_type = if is_idr {
            vkn::StdVideoH265SliceType_STD_VIDEO_H265_SLICE_TYPE_I
        } else {
            vkn::StdVideoH265SliceType_STD_VIDEO_H265_SLICE_TYPE_P
        };

        let mut std_slice = vkn::StdVideoEncodeH265SliceSegmentHeader {
            flags: vkn::StdVideoEncodeH265SliceSegmentHeaderFlags {
                _bitfield_align_1: [],
                _bitfield_1:
                    vkn::StdVideoEncodeH265SliceSegmentHeaderFlags::new_bitfield_1(
                        /* first_slice_segment_in_pic_flag */ 1,
                        /* dependent_slice_segment_flag    */ 0,
                        /* slice_sao_luma_flag             */ 0,
                        /* slice_sao_chroma_flag           */ 0,
                        /* num_ref_idx_active_override     */ 0,
                        /* mvd_l1_zero_flag                */ 0,
                        /* cabac_init_flag                 */ 0,
                        /* cu_chroma_qp_offset_enabled     */ 0,
                        /* deblocking_filter_override_flag */ 0,
                        /* slice_deblocking_filter_disabled*/ 0,
                        /* collocated_from_l0_flag         */ if is_idr { 0 } else { 1 },
                        /* slice_loop_filter_across_slices */ 1,
                        /* reserved                        */ 0,
                    ),
            },
            slice_type,
            slice_segment_address: 0,
            collocated_ref_idx: 0,
            MaxNumMergeCand: 5,
            slice_cb_qp_offset: 0,
            slice_cr_qp_offset: 0,
            slice_beta_offset_div2: 0,
            slice_tc_offset_div2: 0,
            slice_act_y_qp_offset: 0,
            slice_act_cb_qp_offset: 0,
            slice_act_cr_qp_offset: 0,
            slice_qp_delta: 0,
            reserved1: 0,
            pWeightTable: std::ptr::null(),
        };
        // Suppress unused-mut warning if the compiler can't see through the
        // raw struct construction.
        let _ = &mut std_slice;

        let ref_lists = vkn::StdVideoEncodeH265ReferenceListsInfo {
            flags: vkn::StdVideoEncodeH265ReferenceListsInfoFlags {
                _bitfield_align_1: [],
                _bitfield_1:
                    vkn::StdVideoEncodeH265ReferenceListsInfoFlags::new_bitfield_1(0, 0, 0),
            },
            num_ref_idx_l0_active_minus1: 0,
            num_ref_idx_l1_active_minus1: 0,
            // 0xff is the spec-mandated "unused" sentinel for the lists.
            RefPicList0: {
                let mut a = [0xffu8; 15];
                if let Some(prev) = ref_slot {
                    // Must match slot_index of pReferenceSlots[0] in encode_info.
                    a[0] = prev as u8;
                }
                a
            },
            RefPicList1: [0xffu8; 15],
            list_entry_l0: [0u8; 15],
            list_entry_l1: [0u8; 15],
        };

        let pic_flags_bf = vkn::StdVideoEncodeH265PictureInfoFlags::new_bitfield_1(
            /* is_reference                */ 1,
            /* IrapPicFlag                 */ if is_idr { 1 } else { 0 },
            /* used_for_long_term_reference*/ 0,
            /* discardable_flag            */ 0,
            /* cross_layer_bla_flag        */ 0,
            /* pic_output_flag             */ 1,
            /* no_output_of_prior_pics_flag*/ 0,
            /* short_term_ref_pic_set_sps_flag*/ if is_idr { 0 } else { 1 },
            /* slice_temporal_mvp_enabled_flag*/ 0,
            /* reserved                    */ 0,
        );

        let std_pic_info = vkn::StdVideoEncodeH265PictureInfo {
            flags: vkn::StdVideoEncodeH265PictureInfoFlags {
                _bitfield_align_1: [],
                _bitfield_1: pic_flags_bf,
            },
            pic_type,
            sps_video_parameter_set_id: VPS_ID,
            pps_seq_parameter_set_id: SPS_ID,
            pps_pic_parameter_set_id: PPS_ID,
            short_term_ref_pic_set_idx: 0,
            PicOrderCntVal: pic_order_cnt,
            TemporalId: 0,
            reserved1: [0; 7],
            pRefLists: &ref_lists,
            pShortTermRefPicSet: std::ptr::null(),
            pLongTermRefPics: std::ptr::null(),
        };

        let nalu_slice = vk::VideoEncodeH265NaluSliceSegmentInfoKHR::default()
            .constant_qp(self.params_qp())
            .std_slice_segment_header(&std_slice);

        let h265_picture_info = vk::VideoEncodeH265PictureInfoKHR::default()
            .nalu_slice_segment_entries(std::slice::from_ref(&nalu_slice))
            .std_picture_info(&std_pic_info);

        // === Reference setup & reference slot infos for vkCmdBeginVideoCoding ===
        let setup_std_ref = vkn::StdVideoEncodeH265ReferenceInfo {
            flags: vkn::StdVideoEncodeH265ReferenceInfoFlags {
                _bitfield_align_1: [],
                _bitfield_1:
                    vkn::StdVideoEncodeH265ReferenceInfoFlags::new_bitfield_1(0, 0, 0),
            },
            pic_type,
            PicOrderCntVal: pic_order_cnt,
            TemporalId: 0,
        };
        let setup_h265 = vk::VideoEncodeH265DpbSlotInfoKHR::default()
            .std_reference_info(&setup_std_ref);

        let setup_pic_resource = vk::VideoPictureResourceInfoKHR::default()
            .coded_offset(vk::Offset2D { x: 0, y: 0 })
            .coded_extent(vk::Extent2D {
                width: self.params.width,
                height: self.params.height,
            })
            .base_array_layer(0)
            .image_view_binding(self.dpb[setup_slot].view);

        let mut setup_h265_slot = setup_h265;
        let setup_ref_info = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(setup_slot as i32)
            .picture_resource(&setup_pic_resource)
            .push(&mut setup_h265_slot);

        // The reference slot (for P frames): same shape but with a slot_index
        // matching the previously-set-up DPB.
        let ref_std_ref;
        let ref_h265;
        let ref_pic_resource;
        let mut ref_h265_slot;
        let reference_slots_storage: Option<vk::VideoReferenceSlotInfoKHR<'_>>;

        if let Some(prev) = ref_slot {
            ref_std_ref = vkn::StdVideoEncodeH265ReferenceInfo {
                flags: vkn::StdVideoEncodeH265ReferenceInfoFlags {
                    _bitfield_align_1: [],
                    _bitfield_1:
                        vkn::StdVideoEncodeH265ReferenceInfoFlags::new_bitfield_1(0, 0, 0),
                },
                pic_type: vkn::StdVideoH265PictureType_STD_VIDEO_H265_PICTURE_TYPE_P,
                PicOrderCntVal: (self.frame_index as i32) - 1,
                TemporalId: 0,
            };
            ref_h265 = vk::VideoEncodeH265DpbSlotInfoKHR::default()
                .std_reference_info(&ref_std_ref);
            ref_h265_slot = ref_h265;
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
                    .push(&mut ref_h265_slot),
            );
        } else {
            ref_std_ref = unsafe { std::mem::zeroed() };
            ref_h265 = vk::VideoEncodeH265DpbSlotInfoKHR::default();
            ref_h265_slot = ref_h265;
            ref_pic_resource = vk::VideoPictureResourceInfoKHR::default();
            reference_slots_storage = None;
        }

        // The Begin call needs BOTH the setup slot (with picture_resource =
        // null on slot but with slot_index ≥ 0 — actually no, per spec the
        // setup slot in BeginCoding has picture_resource = nullptr and is the
        // reservation for the slot we're about to write). We assemble the full
        // begin_slots array now.

        let begin_setup_pic_resource_holder; // keep alive
        let begin_setup_h265_holder;
        let mut begin_slots: Vec<vk::VideoReferenceSlotInfoKHR<'_>> = Vec::new();

        // The setup slot in Begin: slot_index = setup_slot, picture_resource =
        // null (we're declaring intent, not binding yet).
        begin_setup_h265_holder = vk::VideoEncodeH265DpbSlotInfoKHR::default();
        begin_setup_pic_resource_holder = vk::VideoPictureResourceInfoKHR::default()
            .coded_offset(vk::Offset2D { x: 0, y: 0 })
            .coded_extent(vk::Extent2D {
                width: self.params.width,
                height: self.params.height,
            })
            .base_array_layer(0)
            .image_view_binding(self.dpb[setup_slot].view);
        // VkVideoEncodeH265DpbSlotInfoKHR REQUIRES non-null pStdReferenceInfo
        // even for the "to-be-setup" slot in BeginCoding.
        let begin_setup_std_ref = vkn::StdVideoEncodeH265ReferenceInfo {
            flags: vkn::StdVideoEncodeH265ReferenceInfoFlags {
                _bitfield_align_1: [],
                _bitfield_1: vkn::StdVideoEncodeH265ReferenceInfoFlags::new_bitfield_1(0, 0, 0),
            },
            pic_type,
            PicOrderCntVal: pic_order_cnt,
            TemporalId: 0,
        };
        let begin_setup_h265 = vk::VideoEncodeH265DpbSlotInfoKHR::default()
            .std_reference_info(&begin_setup_std_ref);
        let mut begin_setup_h265 = begin_setup_h265;
        let _ = begin_setup_h265_holder;
        // Setup slot in BeginCoding uses slot_index = -1 because the slot
        // becomes active only after the encode op writes to it.
        let setup_in_begin = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(-1)
            .picture_resource(&begin_setup_pic_resource_holder)
            .push(&mut begin_setup_h265);
        begin_slots.push(setup_in_begin);

        if let Some(ref_info) = &reference_slots_storage {
            begin_slots.push(*ref_info);
        }

        // Begin scope MUST chain VkVideoEncodeRateControlInfoKHR with mode =
        // DISABLED to match the session's persistent RC state set via
        // cmd_control. Without this, the validator (and the spec) treats the
        // scope as DEFAULT mode, which then conflicts with per-slice
        // constant_qp != 0.
        let mut begin_rc_h265 = vk::VideoEncodeH265RateControlInfoKHR::default()
            .flags(vk::VideoEncodeH265RateControlFlagsKHR::REGULAR_GOP)
            .gop_frame_count(u32::MAX)
            .idr_period(u32::MAX)
            .consecutive_b_frame_count(0)
            .sub_layer_count(1);
        let mut begin_rc = vk::VideoEncodeRateControlInfoKHR::default()
            .rate_control_mode(vk::VideoEncodeRateControlModeFlagsKHR::DISABLED);
        let begin_info_with_rc = vk::VideoBeginCodingInfoKHR::default()
            .video_session(self.session)
            .video_session_parameters(self.session_params)
            .reference_slots(&begin_slots)
            .push(&mut begin_rc)
            .push(&mut begin_rc_h265);

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

        // RC was initialized once at session creation; no per-frame control needed.

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

        let mut h265_picture_info_mut = h265_picture_info;
        let mut encode_info = vk::VideoEncodeInfoKHR::default()
            .dst_buffer(self.out_buffer)
            .dst_buffer_offset(0)
            .dst_buffer_range(OUTPUT_BUFFER_BYTES)
            .src_picture_resource(src_pic_resource)
            .setup_reference_slot(&setup_ref_info)
            .push(&mut h265_picture_info_mut);

        let ref_for_encode_storage;
        if let Some(prev) = ref_slot {
            // Reuse the ref_pic_resource + ref_h265_slot we already filled in.
            ref_for_encode_storage = vk::VideoReferenceSlotInfoKHR::default()
                .slot_index(prev as i32)
                .picture_resource(&ref_pic_resource)
                .push(&mut ref_h265_slot);
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
        // Query has 2 u32 result components (BITSTREAM_BYTES_WRITTEN +
        // BITSTREAM_BUFFER_OFFSET). We want 1 query × 2 components — package
        // as [[u32; 2]; 1] so ash uses queryCount=1 and stride=8 (sizeof of
        // the per-element type), not queryCount=2 and stride=4.
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
        // Feedback components are returned in bit-position order: BUFFER_OFFSET
        // (bit 0) then BYTES_WRITTEN (bit 1).
        let buffer_offset = feedback[0][0] as u64;
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

        // Annex-B framing: driver already emits start codes per Vulkan Video
        // spec when the encode H.265 extension is enabled with default config.

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

impl H265Encoder {
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

impl Drop for H265Encoder {
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
        spec_version: STD_H265_ENCODE_API_VERSION_1_0_0,
    };
    let name = STD_H265_ENCODE_EXTENSION_NAME.to_bytes_with_nul();
    for (i, &b) in name.iter().enumerate() {
        props.extension_name[i] = b as c_char;
    }
    props
}

struct VpsStorage {
    vps: vkn::StdVideoH265VideoParameterSet,
    _ptl: Box<vkn::StdVideoH265ProfileTierLevel>,
    _dpbm: Box<vkn::StdVideoH265DecPicBufMgr>,
}

fn build_vps() -> VpsStorage {
    let ptl = Box::new(vkn::StdVideoH265ProfileTierLevel {
        flags: vkn::StdVideoH265ProfileTierLevelFlags {
            _bitfield_align_1: [],
            _bitfield_1: vkn::StdVideoH265ProfileTierLevelFlags::new_bitfield_1(
                /* general_tier_flag                  */ 0,
                /* general_progressive_source_flag    */ 1,
                /* general_interlaced_source_flag     */ 0,
                /* general_non_packed_constraint_flag */ 1,
                /* general_frame_only_constraint_flag */ 1,
            ),
            __bindgen_padding_0: [0; 3],
        },
        general_profile_idc: vkn::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN,
        general_level_idc: vkn::StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_4_0,
    });
    let dpbm = Box::new(zeroed_dpb_buf_mgr());

    let vps = vkn::StdVideoH265VideoParameterSet {
        flags: vkn::StdVideoH265VpsFlags {
            _bitfield_align_1: [],
            _bitfield_1: vkn::StdVideoH265VpsFlags::new_bitfield_1(
                /* vps_temporal_id_nesting_flag           */ 1,
                /* vps_sub_layer_ordering_info_present    */ 0,
                /* vps_timing_info_present_flag           */ 0,
                /* vps_poc_proportional_to_timing_flag    */ 0,
            ),
            __bindgen_padding_0: [0; 3],
        },
        vps_video_parameter_set_id: VPS_ID,
        vps_max_sub_layers_minus1: 0,
        reserved1: 0,
        reserved2: 0,
        vps_num_units_in_tick: 1,
        vps_time_scale: 60,
        vps_num_ticks_poc_diff_one_minus1: 0,
        reserved3: 0,
        pDecPicBufMgr: &*dpbm,
        pHrdParameters: std::ptr::null(),
        pProfileTierLevel: &*ptl,
    };

    VpsStorage {
        vps,
        _ptl: ptl,
        _dpbm: dpbm,
    }
}

struct SpsStorage {
    sps: vkn::StdVideoH265SequenceParameterSet,
    _ptl: Box<vkn::StdVideoH265ProfileTierLevel>,
    _dpbm: Box<vkn::StdVideoH265DecPicBufMgr>,
    _strps: Box<vkn::StdVideoH265ShortTermRefPicSet>,
}

fn build_sps(width: u32, height: u32) -> SpsStorage {
    let ptl = Box::new(vkn::StdVideoH265ProfileTierLevel {
        flags: vkn::StdVideoH265ProfileTierLevelFlags {
            _bitfield_align_1: [],
            _bitfield_1: vkn::StdVideoH265ProfileTierLevelFlags::new_bitfield_1(
                0, 1, 0, 1, 1,
            ),
            __bindgen_padding_0: [0; 3],
        },
        general_profile_idc: vkn::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN,
        general_level_idc: vkn::StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_4_0,
    });
    let dpbm = Box::new(zeroed_dpb_buf_mgr());

    // Single short-term ref pic set: one negative delta-POC (the prev P/IDR).
    let strps = Box::new(vkn::StdVideoH265ShortTermRefPicSet {
        flags: vkn::StdVideoH265ShortTermRefPicSetFlags {
            _bitfield_align_1: [],
            _bitfield_1: vkn::StdVideoH265ShortTermRefPicSetFlags::new_bitfield_1(0, 0),
            __bindgen_padding_0: [0; 3],
        },
        delta_idx_minus1: 0,
        use_delta_flag: 0,
        abs_delta_rps_minus1: 0,
        used_by_curr_pic_flag: 0x0001,
        used_by_curr_pic_s0_flag: 0x0001,
        used_by_curr_pic_s1_flag: 0,
        reserved1: 0,
        reserved2: 0,
        reserved3: 0,
        num_negative_pics: 1,
        num_positive_pics: 0,
        delta_poc_s0_minus1: [0u16; 16],
        delta_poc_s1_minus1: [0u16; 16],
    });

    let sps = vkn::StdVideoH265SequenceParameterSet {
        flags: vkn::StdVideoH265SpsFlags {
            _bitfield_align_1: [],
            _bitfield_1: vkn::StdVideoH265SpsFlags::new_bitfield_1(
                /* sps_temporal_id_nesting_flag           */ 1,
                /* separate_colour_plane_flag             */ 0,
                /* conformance_window_flag                */ 0,
                /* sps_sub_layer_ordering_info_present    */ 0,
                /* scaling_list_enabled_flag              */ 0,
                /* sps_scaling_list_data_present_flag     */ 0,
                /* amp_enabled_flag                       */ 1,
                /* sample_adaptive_offset_enabled_flag    */ 0,
                /* pcm_enabled_flag                       */ 0,
                /* pcm_loop_filter_disabled_flag          */ 0,
                /* long_term_ref_pics_present_flag        */ 0,
                /* sps_temporal_mvp_enabled_flag          */ 0,
                /* strong_intra_smoothing_enabled_flag    */ 1,
                /* vui_parameters_present_flag            */ 0,
                /* sps_extension_present_flag             */ 0,
                /* sps_range_extension_flag               */ 0,
                /* transform_skip_rotation_enabled_flag   */ 0,
                /* transform_skip_context_enabled_flag    */ 0,
                /* implicit_rdpcm_enabled_flag            */ 0,
                /* explicit_rdpcm_enabled_flag            */ 0,
                /* extended_precision_processing_flag     */ 0,
                /* intra_smoothing_disabled_flag          */ 0,
                /* high_precision_offsets_enabled_flag    */ 0,
                /* persistent_rice_adaptation_enabled_flag*/ 0,
                /* cabac_bypass_alignment_enabled_flag    */ 0,
                /* sps_scc_extension_flag                 */ 0,
                /* sps_curr_pic_ref_enabled_flag          */ 0,
                /* palette_mode_enabled_flag              */ 0,
                /* sps_palette_predictor_initializers_pf  */ 0,
                /* intra_boundary_filtering_disabled_flag */ 0,
            ),
        },
        chroma_format_idc: vkn::StdVideoH265ChromaFormatIdc_STD_VIDEO_H265_CHROMA_FORMAT_IDC_420,
        pic_width_in_luma_samples: width,
        pic_height_in_luma_samples: height,
        sps_video_parameter_set_id: VPS_ID,
        sps_max_sub_layers_minus1: 0,
        sps_seq_parameter_set_id: SPS_ID,
        bit_depth_luma_minus8: 0,
        bit_depth_chroma_minus8: 0,
        log2_max_pic_order_cnt_lsb_minus4: 4,
        log2_min_luma_coding_block_size_minus3: 0,
        log2_diff_max_min_luma_coding_block_size: 3,
        log2_min_luma_transform_block_size_minus2: 0,
        log2_diff_max_min_luma_transform_block_size: 3,
        max_transform_hierarchy_depth_inter: 0,
        max_transform_hierarchy_depth_intra: 0,
        num_short_term_ref_pic_sets: 1,
        num_long_term_ref_pics_sps: 0,
        pcm_sample_bit_depth_luma_minus1: 0,
        pcm_sample_bit_depth_chroma_minus1: 0,
        log2_min_pcm_luma_coding_block_size_minus3: 0,
        log2_diff_max_min_pcm_luma_coding_block_size: 0,
        reserved1: 0,
        reserved2: 0,
        palette_max_size: 0,
        delta_palette_max_predictor_size: 0,
        motion_vector_resolution_control_idc: 0,
        sps_num_palette_predictor_initializers_minus1: 0,
        conf_win_left_offset: 0,
        conf_win_right_offset: 0,
        conf_win_top_offset: 0,
        conf_win_bottom_offset: 0,
        pProfileTierLevel: &*ptl,
        pDecPicBufMgr: &*dpbm,
        pScalingLists: std::ptr::null(),
        pShortTermRefPicSet: &*strps,
        pLongTermRefPicsSps: std::ptr::null(),
        pSequenceParameterSetVui: std::ptr::null(),
        pPredictorPaletteEntries: std::ptr::null(),
    };

    SpsStorage {
        sps,
        _ptl: ptl,
        _dpbm: dpbm,
        _strps: strps,
    }
}

struct PpsStorage {
    pps: vkn::StdVideoH265PictureParameterSet,
}

fn build_pps() -> PpsStorage {
    let pps = vkn::StdVideoH265PictureParameterSet {
        flags: vkn::StdVideoH265PpsFlags {
            _bitfield_align_1: [],
            _bitfield_1: vkn::StdVideoH265PpsFlags::new_bitfield_1(
                /* dependent_slice_segments_enabled_flag       */ 0,
                /* output_flag_present_flag                    */ 0,
                /* sign_data_hiding_enabled_flag               */ 0,
                /* cabac_init_present_flag                     */ 1,
                /* constrained_intra_pred_flag                 */ 0,
                /* transform_skip_enabled_flag                 */ 0,
                /* cu_qp_delta_enabled_flag                    */ 0,
                /* pps_slice_chroma_qp_offsets_present_flag    */ 0,
                /* weighted_pred_flag                          */ 0,
                /* weighted_bipred_flag                        */ 0,
                /* transquant_bypass_enabled_flag              */ 0,
                /* tiles_enabled_flag                          */ 0,
                /* entropy_coding_sync_enabled_flag            */ 0,
                /* uniform_spacing_flag                        */ 0,
                /* loop_filter_across_tiles_enabled_flag       */ 1,
                /* pps_loop_filter_across_slices_enabled_flag  */ 1,
                /* deblocking_filter_control_present_flag      */ 1,
                /* deblocking_filter_override_enabled_flag     */ 0,
                /* pps_deblocking_filter_disabled_flag         */ 0,
                /* pps_scaling_list_data_present_flag          */ 0,
                /* lists_modification_present_flag             */ 0,
                /* slice_segment_header_extension_present_flag */ 0,
                /* pps_extension_present_flag                  */ 0,
                /* cross_component_prediction_enabled_flag     */ 0,
                /* chroma_qp_offset_list_enabled_flag          */ 0,
                /* pps_curr_pic_ref_enabled_flag               */ 0,
                /* residual_adaptive_colour_transform_enabled  */ 0,
                /* pps_slice_act_qp_offsets_present_flag       */ 0,
                /* pps_palette_predictor_initializers_present  */ 0,
                /* monochrome_palette_flag                     */ 0,
                /* pps_range_extension_flag                    */ 0,
            ),
        },
        pps_pic_parameter_set_id: PPS_ID,
        pps_seq_parameter_set_id: SPS_ID,
        sps_video_parameter_set_id: VPS_ID,
        num_extra_slice_header_bits: 0,
        num_ref_idx_l0_default_active_minus1: 0,
        num_ref_idx_l1_default_active_minus1: 0,
        init_qp_minus26: 0,
        diff_cu_qp_delta_depth: 0,
        pps_cb_qp_offset: 0,
        pps_cr_qp_offset: 0,
        pps_beta_offset_div2: 0,
        pps_tc_offset_div2: 0,
        log2_parallel_merge_level_minus2: 0,
        log2_max_transform_skip_block_size_minus2: 0,
        diff_cu_chroma_qp_offset_depth: 0,
        chroma_qp_offset_list_len_minus1: 0,
        cb_qp_offset_list: [0; 6],
        cr_qp_offset_list: [0; 6],
        log2_sao_offset_scale_luma: 0,
        log2_sao_offset_scale_chroma: 0,
        pps_act_y_qp_offset_plus5: 0,
        pps_act_cb_qp_offset_plus5: 0,
        pps_act_cr_qp_offset_plus3: 0,
        pps_num_palette_predictor_initializers: 0,
        luma_bit_depth_entry_minus8: 0,
        chroma_bit_depth_entry_minus8: 0,
        num_tile_columns_minus1: 0,
        num_tile_rows_minus1: 0,
        reserved1: 0,
        reserved2: 0,
        column_width_minus1: [0; 19],
        row_height_minus1: [0; 21],
        reserved3: 0,
        pScalingLists: std::ptr::null(),
        pPredictorPaletteEntries: std::ptr::null(),
    };
    PpsStorage { pps }
}

fn zeroed_dpb_buf_mgr() -> vkn::StdVideoH265DecPicBufMgr {
    // DecPicBufMgr is a plain array struct — the spec allows zero
    // initialization when we don't override sub-layer ordering info.
    // SAFETY: All fields are integers (no pointers).
    unsafe { std::mem::zeroed() }
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
    let mut h265_get = vk::VideoEncodeH265SessionParametersGetInfoKHR::default()
        .write_std_vps(true)
        .write_std_sps(true)
        .write_std_pps(true)
        .std_vps_id(VPS_ID as u32)
        .std_sps_id(SPS_ID as u32)
        .std_pps_id(PPS_ID as u32);
    let get_info = vk::VideoEncodeSessionParametersGetInfoKHR::default()
        .video_session_parameters(session_params)
        .push(&mut h265_get);

    let mut feedback_h265 = vk::VideoEncodeH265SessionParametersFeedbackInfoKHR::default();
    let mut feedback = vk::VideoEncodeSessionParametersFeedbackInfoKHR::default()
        .push(&mut feedback_h265);

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
    // SAFETY: driver wrote `len` bytes into the buffer.
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
        // Image with VIDEO_ENCODE_DPB usage + profile list.
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

fn create_encode_query_pool(
    ctx: &VulkanContext,
    profile: &ProfileChain,
) -> Result<vk::QueryPool, EncodeError> {
    let _ = profile;
    // Build the chain bottom-up so each `push` sees a null p_next.
    let mut h265_profile_for_query = vk::VideoEncodeH265ProfileInfoKHR::default()
        .std_profile_idc(vkn::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN);
    let mut profile_for_query = vk::VideoProfileInfoKHR::default()
        .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H265)
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
        .push(&mut h265_profile_for_query)
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

    let mut rc_h265 = vk::VideoEncodeH265RateControlInfoKHR::default()
        .flags(vk::VideoEncodeH265RateControlFlagsKHR::REGULAR_GOP)
        .gop_frame_count(u32::MAX)
        .idr_period(u32::MAX)
        .consecutive_b_frame_count(0)
        .sub_layer_count(1);
    let mut rc = vk::VideoEncodeRateControlInfoKHR::default()
        .rate_control_mode(vk::VideoEncodeRateControlModeFlagsKHR::DISABLED);
    let control = vk::VideoCodingControlInfoKHR::default()
        .flags(
            vk::VideoCodingControlFlagsKHR::RESET
                | vk::VideoCodingControlFlagsKHR::ENCODE_RATE_CONTROL,
        )
        .push(&mut rc)
        .push(&mut rc_h265);
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
    queue_family: u32,
) -> Result<(), EncodeError> {
    let _ = queue_family;
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

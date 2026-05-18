//! AV1 encoder backend using Vulkan Video encode KHR extensions.
//!
//! Mirrors the structural flow of `h265.rs` (session creation, memory binding,
//! DPB pool, encode loop, query feedback, one-shot RC init) but emits an OBU
//! stream instead of NAL units and uses AV1-specific std structures.
//!
//! Pipeline: NV12 `VkImage` → `VkVideoSessionKHR` (AV1) → DPB pool →
//! `vkCmdEncodeVideoKHR` → output buffer → OBU readback. CQP rate control
//! only; single-reference (low-latency) prediction.

#![allow(clippy::too_many_lines)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::undocumented_unsafe_blocks)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_lossless)]
#![allow(non_upper_case_globals)]

use std::ffi::{CStr, c_char};
use std::sync::Arc;

use ash::vk;
use ash::vk::TaggedStructure;
use ash::vk::native as vkn;
use remoteway_vulkan::{VideoCodec, VulkanContext};

use crate::EncodeError;
use crate::encoder::{EncodeParams, EncodedFrame, Encoder, FrameKind, InputFrame, RateControl};

/// Two-slot DPB: one reference (LAST_FRAME) + one setup (current frame),
/// the bare minimum for an KEY-then-INTER-only low-latency loop.
const DPB_SLOTS: u32 = 2;
/// `VK_MAKE_VIDEO_STD_VERSION(1, 0, 0)`.
const STD_AV1_ENCODE_API_VERSION_1_0_0: u32 = 1 << 22;
const STD_AV1_ENCODE_EXTENSION_NAME: &CStr = c"VK_STD_vulkan_video_codec_av1_encode";

/// NV12 picture format consumed by the encoder.
const INPUT_FORMAT: vk::Format = vk::Format::G8_B8R8_2PLANE_420_UNORM;

/// Reasonable upper bound for one 720p frame's encoded payload.
const OUTPUT_BUFFER_BYTES: u64 = 1024 * 1024;

/// AV1 `STD_VIDEO_AV1_PRIMARY_REF_NONE` (sentinel used in `primary_ref_frame`
/// for KEY frames — indicates no primary reference). The C header defines
/// this as `7`.
const STD_VIDEO_AV1_PRIMARY_REF_NONE: u8 = 7;

/// Index in `referenceNameSlotIndices` for LAST_FRAME. AV1's seven reference
/// names map to indices 0..7 in this Vulkan array.
const REF_NAME_LAST_FRAME: usize = 0;

/// AV1 spec: `refresh_frame_flags` is an 8-bit mask of which of the 8 AV1
/// saved-frame buffer slots get updated by this frame. We always refresh
/// slot 0 (writing into LAST_FRAME), keeping the single-reference loop
/// simple.
const REFRESH_LAST_FRAME_ONLY: u8 = 0x01;

pub struct Av1Encoder {
    ctx: Arc<VulkanContext>,
    params: EncodeParams,
    encode_queue_family: u32,
    encode_queue: vk::Queue,

    video_queue: ash::khr::video_queue::Device,
    video_encode: ash::khr::video_encode_queue::Device,

    profile: ProfileChain,

    session: vk::VideoSessionKHR,
    session_memory: Vec<vk::DeviceMemory>,
    session_params: vk::VideoSessionParametersKHR,

    /// Sequence header OBU bytes from `vkGetEncodedVideoSessionParametersKHR`.
    /// Re-emitted in [`EncodedFrame::parameter_sets`] on every KEY frame so a
    /// receiver who joins at a keyframe can decode.
    parameter_sets_blob: Vec<u8>,

    dpb: Vec<DpbSlot>,
    /// Index of the slot the previous KEY/INTER was written to. Next frame
    /// uses this as its LAST_FRAME reference; the new frame is written to
    /// the *other* slot. `None` until the first frame has been encoded.
    prior_ref_slot: Option<usize>,

    out_buffer: vk::Buffer,
    out_buffer_memory: vk::DeviceMemory,

    /// 1 query × 2 u32 components per frame: BITSTREAM_BUFFER_OFFSET (bit 0)
    /// and BITSTREAM_BYTES_WRITTEN (bit 1).
    query_pool: vk::QueryPool,

    encode_cmd_pool: vk::CommandPool,
    encode_cmd: vk::CommandBuffer,
    encode_fence: vk::Fence,

    frame_index: u64,
    /// AV1 per-frame counter (used for order_hint, wraps at 2^OrderHintBits).
    /// We use the default 8-bit order_hint, so wraps at 256.
    order_hint: u8,
    force_keyframe: bool,

    /// Cache of `VkImageView` per input `VkImage` to avoid leaking views on
    /// every `encode()`. Destroyed in `Drop`.
    input_views: std::collections::HashMap<vk::Image, vk::ImageView>,
}

struct DpbSlot {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
}

/// Owns the AV1 profile structs so their pointers stay valid for the lifetime
/// of the encoder (the driver re-reads them on every `vkCmdBeginVideoCodingKHR`
/// via the video session's stored profile).
struct ProfileChain {
    av1: Box<vk::VideoEncodeAV1ProfileInfoKHR<'static>>,
    profile: Box<vk::VideoProfileInfoKHR<'static>>,
    profile_list: Box<vk::VideoProfileListInfoKHR<'static>>,
}

impl ProfileChain {
    fn new() -> Self {
        let av1: Box<vk::VideoEncodeAV1ProfileInfoKHR<'static>> = Box::new(
            vk::VideoEncodeAV1ProfileInfoKHR::default()
                .std_profile(vkn::StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_MAIN),
        );
        let mut profile: Box<vk::VideoProfileInfoKHR<'static>> = Box::new(
            vk::VideoProfileInfoKHR::default()
                .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_AV1)
                .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
                .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
                .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8),
        );
        // Pin av1 in the chain. Box address is stable.
        profile.p_next =
            &*av1 as *const vk::VideoEncodeAV1ProfileInfoKHR<'_> as *const std::ffi::c_void;

        let mut profile_list: Box<vk::VideoProfileListInfoKHR<'static>> =
            Box::new(vk::VideoProfileListInfoKHR::default());
        profile_list.profile_count = 1;
        profile_list.p_profiles = &*profile;

        Self {
            av1,
            profile,
            profile_list,
        }
    }

    fn profile(&self) -> &vk::VideoProfileInfoKHR<'static> {
        &self.profile
    }
}

impl Encoder for Av1Encoder {
    fn new(ctx: Arc<VulkanContext>, params: EncodeParams) -> Result<Self, EncodeError> {
        if params.codec != VideoCodec::Av1 {
            return Err(EncodeError::InvalidParams(format!(
                "Av1Encoder cannot encode {:?}",
                params.codec
            )));
        }
        let caps = ctx.probe_video_encode_capabilities(VideoCodec::Av1)?;
        params.validate_against(&caps)?;

        let encode_queue_family = ctx.video_encode_queue_family.ok_or_else(|| {
            EncodeError::InvalidParams("context has no encode queue family".into())
        })?;
        let encode_queue = ctx.video_encode_queue.ok_or_else(|| {
            EncodeError::InvalidParams("context has no encode queue handle".into())
        })?;

        let video_queue = ash::khr::video_queue::Device::load(&ctx.instance, &ctx.device);
        let video_encode = ash::khr::video_encode_queue::Device::load(&ctx.instance, &ctx.device);

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
        let session = unsafe { video_queue.create_video_session(&session_info, None) }
            .map_err(|e| EncodeError::SubmitFailed(format!("create_video_session: {e:?}")))?;

        // -- Bind session memory --
        let session_memory = bind_session_memory(&ctx, &video_queue, session)?;

        // -- Build AV1 sequence header & session parameters --
        let seq = build_seq_header(params.width, params.height);

        let mut av1_params_info = vk::VideoEncodeAV1SessionParametersCreateInfoKHR::default()
            .std_sequence_header(&seq.header);

        let session_params_info = vk::VideoSessionParametersCreateInfoKHR::default()
            .video_session(session)
            .push(&mut av1_params_info);

        let session_params =
            unsafe { video_queue.create_video_session_parameters(&session_params_info, None) }
                .map_err(|e| {
                    EncodeError::SubmitFailed(format!("create_video_session_parameters: {e:?}"))
                })?;

        // -- Pull sequence header OBU blob (cache to prepend on KEY frames) --
        let seq_header_obu =
            fetch_parameter_sets_blob(&video_encode, session_params).map_err(|e| {
                EncodeError::ReadbackFailed(format!("get_encoded_video_session_parameters: {e:?}"))
            })?;
        // Prefix the sequence header OBU with its own OBU_TEMPORAL_DELIMITER
        // so the concatenated stream `parameter_sets + data` is a valid AV1
        // OBU bitstream where each temporal unit begins with a TD. Without
        // this, downstream demuxers (ffmpeg's `obu` demuxer, dav1d) skip
        // OBUs that arrive before the first TD.
        let mut parameter_sets_blob = Vec::with_capacity(seq_header_obu.len() + 2);
        parameter_sets_blob.push(0x12);
        parameter_sets_blob.push(0x00);
        parameter_sets_blob.extend_from_slice(&seq_header_obu);
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

        // -- Transition DPB slots to VIDEO_ENCODE_DPB layout --
        transition_dpb_to_initial_layout(
            &ctx,
            encode_cmd_pool,
            encode_queue,
            &dpb,
            encode_queue_family,
        )?;

        // -- One-shot RC init: Begin → control(RESET + ENCODE_RATE_CONTROL DISABLED) → End.
        initialize_rate_control(
            &ctx,
            &video_queue,
            encode_cmd_pool,
            encode_queue,
            session,
            session_params,
        )?;

        // The driver copies the std sequence header into session params; we
        // can drop `seq` here. Nothing else holds a pointer into it.
        drop(seq);

        Ok(Self {
            ctx,
            params,
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
            order_hint: 0,
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

        let is_key = self.frame_index == 0 || self.force_keyframe;
        self.force_keyframe = false;

        // Pick DPB slots: setup = round-robin; reference = previous setup.
        let setup_slot = match self.prior_ref_slot {
            None => 0,
            Some(prev) => (prev + 1) % self.dpb.len(),
        };
        let ref_slot = if is_key { None } else { self.prior_ref_slot };

        let order_hint = self.order_hint;

        let frame_type = if is_key {
            vkn::StdVideoAV1FrameType_STD_VIDEO_AV1_FRAME_TYPE_KEY
        } else {
            vkn::StdVideoAV1FrameType_STD_VIDEO_AV1_FRAME_TYPE_INTER
        };

        // Build StdVideoEncodeAV1PictureInfo
        // Per AV1 spec / NVidia ref: KEY frames with show_frame=1 implicitly
        // have error_resilient_mode=1 (the bit isn't written to the bitstream
        // but the inferred state matters). Set it explicitly to match.
        let pic_flags_bf = vkn::StdVideoEncodeAV1PictureInfoFlags::new_bitfield_1(
            /* error_resilient_mode             */ if is_key { 1 } else { 0 },
            /* disable_cdf_update               */ 0,
            /* use_superres                     */ 0,
            /* render_and_frame_size_different  */ 0,
            /* allow_screen_content_tools       */ 1,
            /* is_filter_switchable             */ 0,
            /* force_integer_mv                 */ if is_key { 1 } else { 0 },
            /* frame_size_override_flag         */ 0,
            /* buffer_removal_time_present_flag */ 0,
            // allow_intrabc: RADV hardcodes this bit to 0 in the bitstream
            // regardless of what we set here (see radv_video_enc.c). Leave
            // it 0 to match expected behavior.
            /* allow_intrabc                    */
            0,
            /* frame_refs_short_signaling       */ 0,
            /* allow_high_precision_mv          */ 0,
            /* is_motion_mode_switchable        */ 0,
            /* use_ref_frame_mvs                */ 0,
            /* disable_frame_end_update_cdf     */ 0,
            /* allow_warped_motion              */ 0,
            /* reduced_tx_set                   */ 0,
            /* skip_mode_present                */ 0,
            /* delta_q_present                  */ 0,
            /* delta_lf_present                 */ 0,
            /* delta_lf_multi                   */ 0,
            /* segmentation_enabled             */ 0,
            /* segmentation_update_map          */ 0,
            /* segmentation_temporal_update     */ 0,
            /* segmentation_update_data         */ 0,
            /* UsesLr                           */ 0,
            /* usesChromaLr                     */ 0,
            /* show_frame                       */ 1,
            /* showable_frame                   */ if is_key { 0 } else { 1 },
            /* reserved                         */ 0,
        );

        // ref_frame_idx[7]: indices into AV1's 8-element saved-frame buffer.
        // For single-reference INTER, LAST_FRAME points to slot 0 of the AV1
        // saved-frame buffer (where we always write via `refresh_frame_flags`).
        let ref_frame_idx: [i8; 7] = [0i8; 7];

        // ref_order_hint[8]: order_hint of the frame currently in each of
        // the 8 saved-frame buffer slots. We only use slot 0; others zero.
        let mut ref_order_hint = [0u8; 8];
        if ref_slot.is_some() {
            ref_order_hint[0] = self.order_hint.wrapping_sub(1);
        }

        let primary_ref_frame = if is_key {
            STD_VIDEO_AV1_PRIMARY_REF_NONE
        } else {
            0 // LAST_FRAME
        };

        // RADV (VCN5+) dereferences `pic->pCDEF` directly when
        // `seq->flags.enable_cdef = 1`. Provide a reasonable CDEF struct
        // (values taken from NVidia's vk_video_samples reference encoder).
        let cdef = vkn::StdVideoAV1CDEF {
            cdef_damping_minus_3: 2,
            cdef_bits: 2,
            cdef_y_pri_strength: [0, 2, 4, 9, 0, 0, 0, 0],
            cdef_y_sec_strength: [0, 0, 0, 0, 0, 0, 0, 0],
            cdef_uv_pri_strength: [0, 0, 0, 0, 0, 0, 0, 0],
            cdef_uv_sec_strength: [0, 0, 0, 0, 0, 0, 0, 0],
        };

        let std_pic_info = vkn::StdVideoEncodeAV1PictureInfo {
            flags: vkn::StdVideoEncodeAV1PictureInfoFlags {
                _bitfield_align_1: [],
                _bitfield_1: pic_flags_bf,
            },
            frame_type,
            frame_presentation_time: 0,
            current_frame_id: 0,
            order_hint,
            primary_ref_frame,
            // refresh_frame_flags: which of AV1's 8 saved-frame buffer slots
            // get overwritten by this frame. KEY frames MUST refresh all 8
            // slots per AV1 spec (5.9.1: allFrames = 0xff). INTER frames in
            // single-reference low-latency mode refresh just slot 0.
            refresh_frame_flags: if is_key {
                0xff
            } else {
                REFRESH_LAST_FRAME_ONLY
            },
            coded_denom: 0,
            render_width_minus_1: (self.params.width - 1) as u16,
            render_height_minus_1: (self.params.height - 1) as u16,
            interpolation_filter:
                vkn::StdVideoAV1InterpolationFilter_STD_VIDEO_AV1_INTERPOLATION_FILTER_EIGHTTAP,
            TxMode: vkn::StdVideoAV1TxMode_STD_VIDEO_AV1_TX_MODE_SELECT,
            delta_q_res: 0,
            delta_lf_res: 0,
            ref_order_hint,
            ref_frame_idx,
            reserved1: [0; 3],
            delta_frame_id_minus_1: [0u32; 7],
            pTileInfo: std::ptr::null(),
            pQuantization: std::ptr::null(),
            pSegmentation: std::ptr::null(),
            pLoopFilter: std::ptr::null(),
            pCDEF: std::ptr::null(),
            pLoopRestoration: std::ptr::null(),
            pGlobalMotion: std::ptr::null(),
            pExtensionHeader: std::ptr::null(),
            pBufferRemovalTimes: std::ptr::null(),
        };

        // referenceNameSlotIndices: 7-element array mapping AV1 reference
        // names (LAST/LAST2/LAST3/GOLDEN/BWDREF/ALTREF2/ALTREF) to DPB slot
        // indices. For SINGLE_REFERENCE INTER, only LAST_FRAME is populated.
        let mut ref_name_slots: [i32; vk::MAX_VIDEO_AV1_REFERENCES_PER_FRAME_KHR] =
            [-1i32; vk::MAX_VIDEO_AV1_REFERENCES_PER_FRAME_KHR];
        if let Some(prev) = ref_slot {
            ref_name_slots[REF_NAME_LAST_FRAME] = prev as i32;
        }

        let prediction_mode = if is_key {
            vk::VideoEncodeAV1PredictionModeKHR::INTRA_ONLY
        } else {
            vk::VideoEncodeAV1PredictionModeKHR::SINGLE_REFERENCE
        };
        let rc_group = if is_key {
            vk::VideoEncodeAV1RateControlGroupKHR::INTRA
        } else {
            vk::VideoEncodeAV1RateControlGroupKHR::PREDICTIVE
        };

        let av1_picture_info = vk::VideoEncodeAV1PictureInfoKHR::default()
            .prediction_mode(prediction_mode)
            .rate_control_group(rc_group)
            .constant_q_index(self.params_q_index())
            .std_picture_info(&std_pic_info)
            .reference_name_slot_indices(ref_name_slots);

        // === Reference setup & reference slot infos for vkCmdBeginVideoCoding ===
        let setup_std_ref = vkn::StdVideoEncodeAV1ReferenceInfo {
            flags: vkn::StdVideoEncodeAV1ReferenceInfoFlags {
                _bitfield_align_1: [],
                _bitfield_1: vkn::StdVideoEncodeAV1ReferenceInfoFlags::new_bitfield_1(0, 0, 0),
            },
            RefFrameId: 0,
            frame_type,
            OrderHint: order_hint,
            reserved1: [0; 3],
            pExtensionHeader: std::ptr::null(),
        };
        let setup_av1_slot =
            vk::VideoEncodeAV1DpbSlotInfoKHR::default().std_reference_info(&setup_std_ref);

        let setup_pic_resource = vk::VideoPictureResourceInfoKHR::default()
            .coded_offset(vk::Offset2D { x: 0, y: 0 })
            .coded_extent(vk::Extent2D {
                width: self.params.width,
                height: self.params.height,
            })
            .base_array_layer(0)
            .image_view_binding(self.dpb[setup_slot].view);

        let mut setup_av1_slot_mut = setup_av1_slot;
        let setup_ref_info = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(setup_slot as i32)
            .picture_resource(&setup_pic_resource)
            .push(&mut setup_av1_slot_mut);

        // The reference slot (for INTER frames): same shape but with a
        // slot_index matching the previously-set-up DPB.
        let ref_std_ref;
        let ref_av1;
        let ref_pic_resource;
        let mut ref_av1_slot;
        let reference_slots_storage: Option<vk::VideoReferenceSlotInfoKHR<'_>>;

        if let Some(prev) = ref_slot {
            ref_std_ref = vkn::StdVideoEncodeAV1ReferenceInfo {
                flags: vkn::StdVideoEncodeAV1ReferenceInfoFlags {
                    _bitfield_align_1: [],
                    _bitfield_1: vkn::StdVideoEncodeAV1ReferenceInfoFlags::new_bitfield_1(0, 0, 0),
                },
                RefFrameId: 0,
                frame_type: vkn::StdVideoAV1FrameType_STD_VIDEO_AV1_FRAME_TYPE_INTER,
                OrderHint: order_hint.wrapping_sub(1),
                reserved1: [0; 3],
                pExtensionHeader: std::ptr::null(),
            };
            ref_av1 = vk::VideoEncodeAV1DpbSlotInfoKHR::default().std_reference_info(&ref_std_ref);
            ref_av1_slot = ref_av1;
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
                    .push(&mut ref_av1_slot),
            );
        } else {
            ref_std_ref = unsafe { std::mem::zeroed() };
            ref_av1 = vk::VideoEncodeAV1DpbSlotInfoKHR::default();
            ref_av1_slot = ref_av1;
            ref_pic_resource = vk::VideoPictureResourceInfoKHR::default();
            reference_slots_storage = None;
        }

        // BeginCoding needs BOTH the setup slot (with slot_index = -1, the
        // reservation for the slot we're about to write) AND the currently-
        // active reference, if any.
        let mut begin_slots: Vec<vk::VideoReferenceSlotInfoKHR<'_>> = Vec::new();

        let begin_setup_pic_resource = vk::VideoPictureResourceInfoKHR::default()
            .coded_offset(vk::Offset2D { x: 0, y: 0 })
            .coded_extent(vk::Extent2D {
                width: self.params.width,
                height: self.params.height,
            })
            .base_array_layer(0)
            .image_view_binding(self.dpb[setup_slot].view);

        // VkVideoEncodeAV1DpbSlotInfoKHR REQUIRES non-null pStdReferenceInfo
        // even for the "to-be-setup" slot in BeginCoding.
        let begin_setup_std_ref = vkn::StdVideoEncodeAV1ReferenceInfo {
            flags: vkn::StdVideoEncodeAV1ReferenceInfoFlags {
                _bitfield_align_1: [],
                _bitfield_1: vkn::StdVideoEncodeAV1ReferenceInfoFlags::new_bitfield_1(0, 0, 0),
            },
            RefFrameId: 0,
            frame_type,
            OrderHint: order_hint,
            reserved1: [0; 3],
            pExtensionHeader: std::ptr::null(),
        };
        let mut begin_setup_av1 =
            vk::VideoEncodeAV1DpbSlotInfoKHR::default().std_reference_info(&begin_setup_std_ref);
        let setup_in_begin = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(-1)
            .picture_resource(&begin_setup_pic_resource)
            .push(&mut begin_setup_av1);
        begin_slots.push(setup_in_begin);

        if let Some(ref_info) = &reference_slots_storage {
            begin_slots.push(*ref_info);
        }

        // Begin scope MUST chain VkVideoEncodeRateControlInfoKHR with mode =
        // DISABLED to match the session's persistent RC state set via
        // cmd_control. AV1 also chains VkVideoEncodeAV1RateControlInfoKHR.
        let mut begin_rc_av1 = vk::VideoEncodeAV1RateControlInfoKHR::default()
            .gop_frame_count(u32::MAX)
            .key_frame_period(u32::MAX)
            .consecutive_bipredictive_frame_count(0)
            .temporal_layer_count(1);
        let mut begin_rc = vk::VideoEncodeRateControlInfoKHR::default()
            .rate_control_mode(vk::VideoEncodeRateControlModeFlagsKHR::DISABLED);
        let begin_info_with_rc = vk::VideoBeginCodingInfoKHR::default()
            .video_session(self.session)
            .video_session_parameters(self.session_params)
            .reference_slots(&begin_slots)
            .push(&mut begin_rc)
            .push(&mut begin_rc_av1);

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

        unsafe {
            self.ctx
                .device
                .cmd_reset_query_pool(self.encode_cmd, self.query_pool, 0, 1);
        }

        unsafe {
            self.video_queue
                .cmd_begin_video_coding(self.encode_cmd, &begin_info_with_rc);
        }

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
            let v = create_image_view(
                &self.ctx,
                frame.image,
                INPUT_FORMAT,
                vk::ImageAspectFlags::COLOR,
            )?;
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

        let mut av1_picture_info_mut = av1_picture_info;
        let mut encode_info = vk::VideoEncodeInfoKHR::default()
            .dst_buffer(self.out_buffer)
            .dst_buffer_offset(0)
            .dst_buffer_range(OUTPUT_BUFFER_BYTES)
            .src_picture_resource(src_pic_resource)
            .setup_reference_slot(&setup_ref_info)
            .push(&mut av1_picture_info_mut);

        let ref_for_encode_storage;
        if let Some(prev) = ref_slot {
            ref_for_encode_storage = vk::VideoReferenceSlotInfoKHR::default()
                .slot_index(prev as i32)
                .picture_resource(&ref_pic_resource)
                .push(&mut ref_av1_slot);
            encode_info =
                encode_info.reference_slots(std::slice::from_ref(&ref_for_encode_storage));
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
            let submit =
                vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&self.encode_cmd));
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
        // 1 query × 2 components = [[u32; 2]; 1]. Components are returned in
        // bit-position order: BUFFER_OFFSET (bit 0), BYTES_WRITTEN (bit 1).
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
                .map_err(|e| {
                    EncodeError::ReadbackFailed(format!("get_query_pool_results: {e:?}"))
                })?;
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

        // Every AV1 temporal unit must begin with an OBU_TEMPORAL_DELIMITER.
        // Driver emits the frame OBU + (optional) tile group OBUs, but does
        // NOT prepend a TD. Prepend `0x12 0x00` here for every frame so the
        // concatenated stream is a valid AV1 OBU bitstream.
        // - 0x12 = OBU header byte: forbidden(0)|type=2(TD)|ext(0)|has_size(1)|res(0)
        // - 0x00 = LEB128 size: 0 payload bytes
        let mut framed = Vec::with_capacity(bitstream.len() + 2);
        framed.push(0x12);
        framed.push(0x00);
        framed.extend_from_slice(&bitstream);
        let bitstream = framed;

        // Sequence header OBU is returned separately in `parameter_sets` on
        // KEY frames so transport can cache & re-send it on client join.
        let parameter_sets = if is_key {
            Some(self.parameter_sets_blob.clone())
        } else {
            None
        };

        let kind = if is_key { FrameKind::Idr } else { FrameKind::P };

        self.frame_index += 1;
        self.order_hint = self.order_hint.wrapping_add(1);
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

impl Av1Encoder {
    /// Maps the configured rate control to an AV1 Q-index (0..=255). AV1 uses
    /// Q-index, not QP. We map QP (typical 0..=51 range) → Q-index by ×4 so
    /// QP 26 → Q-index 104 (mid-quality). Default 128 when non-CQP is in use.
    fn params_q_index(&self) -> u32 {
        match self.params.rate_control {
            RateControl::ConstantQp { qp } => qp.saturating_mul(4).min(255),
            _ => 128,
        }
    }
}

impl Drop for Av1Encoder {
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
            self.ctx.device.free_memory(self.out_buffer_memory, None);
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
// Helpers — std header / sequence header
// =============================================================================

fn std_header_version() -> vk::ExtensionProperties {
    let mut props = vk::ExtensionProperties {
        extension_name: [0; vk::MAX_EXTENSION_NAME_SIZE],
        spec_version: STD_AV1_ENCODE_API_VERSION_1_0_0,
    };
    let name = STD_AV1_ENCODE_EXTENSION_NAME.to_bytes_with_nul();
    for (i, &b) in name.iter().enumerate() {
        props.extension_name[i] = b as c_char;
    }
    props
}

/// Owns the AV1 sequence header plus the color config it points to. Lifetime
/// only needs to survive until `create_video_session_parameters` returns; the
/// driver copies the header internally.
struct SeqHeaderStorage {
    header: vkn::StdVideoAV1SequenceHeader,
    _color: Box<vkn::StdVideoAV1ColorConfig>,
}

fn build_seq_header(width: u32, height: u32) -> SeqHeaderStorage {
    let color_flags_bf = vkn::StdVideoAV1ColorConfigFlags::new_bitfield_1(
        /* mono_chrome                    */ 0,
        /* color_range                    */ 1, // FULL (PC) range
        /* separate_uv_delta_q            */ 0, /* color_description_present_flag */ 0,
        /* reserved                       */ 0,
    );
    let color = Box::new(vkn::StdVideoAV1ColorConfig {
        flags: vkn::StdVideoAV1ColorConfigFlags {
            _bitfield_align_1: [],
            _bitfield_1: color_flags_bf,
        },
        BitDepth: 8,
        subsampling_x: 1,
        subsampling_y: 1,
        reserved1: 0,
        color_primaries: vkn::StdVideoAV1ColorPrimaries_STD_VIDEO_AV1_COLOR_PRIMARIES_BT_709,
        transfer_characteristics:
            vkn::StdVideoAV1TransferCharacteristics_STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_UNSPECIFIED,
        matrix_coefficients:
            vkn::StdVideoAV1MatrixCoefficients_STD_VIDEO_AV1_MATRIX_COEFFICIENTS_UNSPECIFIED,
        chroma_sample_position:
            vkn::StdVideoAV1ChromaSamplePosition_STD_VIDEO_AV1_CHROMA_SAMPLE_POSITION_UNKNOWN,
    });

    let flags_bf = vkn::StdVideoAV1SequenceHeaderFlags::new_bitfield_1(
        /* still_picture                    */ 0,
        /* reduced_still_picture_header     */ 0,
        /* use_128x128_superblock           */ 0, // 64x64 superblocks (RADV-friendly)
        /* enable_filter_intra              */ 1,
        /* enable_intra_edge_filter         */ 1,
        /* enable_interintra_compound       */ 0,
        /* enable_masked_compound           */ 0,
        /* enable_warped_motion             */ 0,
        /* enable_dual_filter               */ 0,
        /* enable_order_hint                */ 1,
        /* enable_jnt_comp                  */ 0,
        /* enable_ref_frame_mvs             */ 0,
        /* frame_id_numbers_present_flag    */ 0,
        /* enable_superres                  */ 0,
        /* enable_cdef                      */ 0,
        /* enable_restoration               */ 0, // RADV forces 0; don't fight it
        /* film_grain_params_present        */ 0,
        /* timing_info_present_flag         */ 0,
        /* initial_display_delay_present    */ 0,
        /* reserved                         */ 0,
    );

    let frame_width_bits = bits_needed(width);
    let frame_height_bits = bits_needed(height);

    let header = vkn::StdVideoAV1SequenceHeader {
        flags: vkn::StdVideoAV1SequenceHeaderFlags {
            _bitfield_align_1: [],
            _bitfield_1: flags_bf,
        },
        seq_profile: vkn::StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_MAIN,
        frame_width_bits_minus_1: (frame_width_bits - 1) as u8,
        frame_height_bits_minus_1: (frame_height_bits - 1) as u8,
        max_frame_width_minus_1: (width - 1) as u16,
        max_frame_height_minus_1: (height - 1) as u16,
        delta_frame_id_length_minus_2: 0,
        additional_frame_id_length_minus_1: 0,
        order_hint_bits_minus_1: 7, // OrderHintBits = 8
        // STD_VIDEO_AV1_SELECT_INTEGER_MV = 2 and
        // STD_VIDEO_AV1_SELECT_SCREEN_CONTENT_TOOLS = 2 are the SELECT
        // sentinels — per-frame picture-info flags pick the values
        // dynamically. Required for desktop/screen-content compression.
        seq_force_integer_mv: 2,
        seq_force_screen_content_tools: 2,
        reserved1: [0; 5],
        pColorConfig: &*color,
        pTimingInfo: std::ptr::null(),
    };

    SeqHeaderStorage {
        header,
        _color: color,
    }
}

/// Returns the number of bits needed to represent `value` (1..=u32::MAX).
/// AV1 spec: `frame_width_bits_minus_1 = ceil(log2(max_frame_width)) - 1`.
/// For 1920 this returns 11 (so minus_1 = 10).
fn bits_needed(value: u32) -> u32 {
    let v = value.max(1);
    let bits = 32 - (v - 1).leading_zeros();
    bits.max(1)
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
            .map_err(|e| EncodeError::SubmitFailed(format!("bind_video_session_memory: {e:?}")))?;
    }

    Ok(allocations)
}

fn fetch_parameter_sets_blob(
    video_encode: &ash::khr::video_encode_queue::Device,
    session_params: vk::VideoSessionParametersKHR,
) -> Result<Vec<u8>, vk::Result> {
    // AV1 has no codec-specific Get/Feedback structs in ash. The generic
    // VkVideoEncodeSessionParametersGetInfoKHR is sufficient: the driver
    // emits the OBU_SEQUENCE_HEADER for the encoded sequence.
    let get_info = vk::VideoEncodeSessionParametersGetInfoKHR::default()
        .video_session_parameters(session_params);

    let mut feedback = vk::VideoEncodeSessionParametersFeedbackInfoKHR::default();

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
    let mut av1_profile_for_query = vk::VideoEncodeAV1ProfileInfoKHR::default()
        .std_profile(vkn::StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_MAIN);
    let mut profile_for_query = vk::VideoProfileInfoKHR::default()
        .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_AV1)
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
        .push(&mut av1_profile_for_query)
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

    let mut rc_av1 = vk::VideoEncodeAV1RateControlInfoKHR::default()
        .gop_frame_count(u32::MAX)
        .key_frame_period(u32::MAX)
        .consecutive_bipredictive_frame_count(0)
        .temporal_layer_count(1);
    let mut rc = vk::VideoEncodeRateControlInfoKHR::default()
        .rate_control_mode(vk::VideoEncodeRateControlModeFlagsKHR::DISABLED);
    let control = vk::VideoCodingControlInfoKHR::default()
        .flags(
            vk::VideoCodingControlFlagsKHR::RESET
                | vk::VideoCodingControlFlagsKHR::ENCODE_RATE_CONTROL,
        )
        .push(&mut rc)
        .push(&mut rc_av1);
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

#![cfg(all(feature = "gpu-tests", feature = "h264"))]

remoteway_encode::encoder_contract_tests!(
    h264_contract,
    ::remoteway_vulkan::VideoCodec::H264,
    ::remoteway_encode::backends::h264::H264Encoder
);

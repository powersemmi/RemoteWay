#![cfg(all(feature = "gpu-tests", feature = "h265"))]

remoteway_encode::encoder_contract_tests!(
    h265_contract,
    ::remoteway_vulkan::VideoCodec::H265,
    ::remoteway_encode::backends::h265::H265Encoder
);

#![cfg(all(feature = "gpu-tests", feature = "av1"))]

remoteway_encode::encoder_contract_tests!(
    av1_contract,
    ::remoteway_vulkan::VideoCodec::Av1,
    ::remoteway_encode::backends::av1::Av1Encoder
);

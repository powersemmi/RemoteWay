use thiserror::Error;

#[derive(Debug, Error)]
pub enum EncodeError {
    #[error("Vulkan: {0}")]
    Vulkan(#[from] remoteway_vulkan::VulkanError),

    #[error("codec {codec:?} not supported on this device")]
    UnsupportedCodec { codec: remoteway_vulkan::VideoCodec },

    #[error("encode params invalid: {0}")]
    InvalidParams(String),

    #[error("encoder not yet initialised")]
    NotReady,

    #[error("encode submission failed: {0}")]
    SubmitFailed(String),

    #[error("bitstream readback failed: {0}")]
    ReadbackFailed(String),
}

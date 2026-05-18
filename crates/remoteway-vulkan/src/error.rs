use thiserror::Error;

#[derive(Debug, Error)]
pub enum VulkanError {
    #[error("Vulkan loader: {0}")]
    LoaderFailed(String),

    #[error("Vulkan init: {0}")]
    InitFailed(String),

    #[error("no Vulkan-capable physical device")]
    NoDevice,

    #[error("no queue family satisfying request: {0}")]
    NoSuitableQueue(String),

    #[error("required extension missing: {0}")]
    MissingExtension(&'static str),

    #[error("Vulkan call failed: {what} ({code:?})")]
    Call {
        what: &'static str,
        code: ash::vk::Result,
    },

    #[error("memory allocation failed: {0}")]
    Allocation(String),
}

impl VulkanError {
    #[allow(dead_code)] // wired up in task #2 (port) and task #3 (encode probes)
    pub(crate) fn call(what: &'static str, code: ash::vk::Result) -> Self {
        Self::Call { what, code }
    }
}

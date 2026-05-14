// Linker stubs for the FidelityFX-SDK Vulkan backend entry points.
//
// The full SDK Vulkan backend (sdk/src/backends/vk/ffx_vk.cpp) is not compiled
// because it pulls in a large amount of shader/dispatch infrastructure that is
// not yet wired into this crate's build.rs. Without these symbols, the static
// archive built from ffx_frameinterpolation.cpp fails to link against the
// Rust crate's FFI declarations in `src/backends/ffx_fg.rs`.
//
// The stubs match the C ABI of the SDK declarations in
// `sdk/include/FidelityFX/host/backends/vk/ffx_vk.h`. They are intentionally
// no-ops returning a generic backend error so that `Fsr3FrameGen::new()`
// fails gracefully at runtime via the existing `InterpolateError::InitFailed`
// path, instead of crashing or silently producing garbage frames.

#include <cstddef>
#include <cstdint>

// FfxErrorCode is an int-sized enum in the SDK; FFX_ERROR_BACKEND_API_ERROR
// from sdk/include/FidelityFX/host/ffx_error.h.
static constexpr int32_t FFX_ERROR_BACKEND_API_ERROR = static_cast<int32_t>(0x8000000d);

extern "C" {

size_t ffxGetScratchMemorySizeVK(void* /*physicalDevice*/, size_t /*maxContexts*/) {
    return 0;
}

int32_t ffxGetInterfaceVK(
    void*  /*backendInterface*/,
    void*  /*device*/,
    void*  /*scratchBuffer*/,
    size_t /*scratchBufferSize*/,
    size_t /*maxContexts*/)
{
    return FFX_ERROR_BACKEND_API_ERROR;
}

} // extern "C"

/// Compile the FidelityFX Frame Interpolation C++ source into a static library.
///
/// Approach: same as `fsr-sys` – use the `cc` crate to compile C++ with GCC/Clang.
/// The SDK source is expected at FIDELITYFX_SDK_DIR or cloned automatically.

use std::path::PathBuf;

fn find_sdk() -> PathBuf {
    if let Ok(d) = std::env::var("FIDELITYFX_SDK_DIR") {
        let p = PathBuf::from(d);
        if p.join("sdk/CMakeLists.txt").exists() { return p.join("sdk"); }
        if p.join("CMakeLists.txt").exists() { return p; }
        panic!("FIDELITYFX_SDK_DIR={} does not contain FidelityFX SDK", p.display());
    }
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let candidates = [
        root.join("../../tmp/FidelityFX-SDK/sdk"),
        root.join("../FidelityFX-SDK/sdk"),
        root.join("FidelityFX-SDK/sdk"),
    ];
    for c in &candidates {
        if c.join("CMakeLists.txt").exists() {
            return c.clone();
        }
    }
    panic!(
        "FidelityFX SDK not found at {}/FidelityFX-SDK/sdk.\n\
         Clone it:\n  \
         git clone --depth 1 --branch v1.1.4 \\\n    \
         https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK.git \\\n    \
         {}/FidelityFX-SDK",
        root.display(),
        root.display()
    );
}

fn main() {
    // Only build when fsr3 feature is enabled.
    if std::env::var("CARGO_FEATURE_FSR3").is_err() {
        return;
    }

    let sdk = find_sdk();
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    // Patch Windows-isms in the source before compiling.
    let fi_cpp = sdk.join("src/components/frameinterpolation/ffx_frameinterpolation.cpp");
    let fi_cpp_patched = out.join("ffx_frameinterpolation_patched.cpp");
    if !fi_cpp_patched.exists() {
        let src = std::fs::read_to_string(&fi_cpp).expect("read ffx_frameinterpolation.cpp");
        let patched = src
            .replace("wcscpy_s", "wcscpy")
            .replace("#include <cfloat>", "#include <cfloat>\n#include <cstring>\n#include <algorithm>");
        std::fs::write(&fi_cpp_patched, patched).expect("write patched ffx_frameinterpolation.cpp");
    }

    // Patch ffx_types.h – replace __declspec(dllexport) with GCC visibility.
    let types_h = sdk.join("include/FidelityFX/host/ffx_types.h");
    let types_h_patched = out.join("ffx_types_patched.h");
    if !types_h_patched.exists() {
        let src = std::fs::read_to_string(&types_h).expect("read ffx_types.h");
        let patched = src.replace(
            "#define FFX_API __declspec(dllexport)",
            "#ifndef FFX_API\n#define FFX_API __attribute__((visibility(\"default\")))\n#endif",
        );
        std::fs::write(&types_h_patched, patched).expect("write patched ffx_types.h");
    }

    let includes = vec![
        out.clone(), // patched headers first
        sdk.join("include"),
        sdk.join("src/shared"),
        sdk.join("include/FidelityFX/host"),
        sdk.join("src/components"),
    ];

    let mut shared_src: Vec<PathBuf> = vec![
        "ffx_assert.cpp",
        "ffx_breadcrumbs_list.cpp",
        "ffx_message.cpp",
        "ffx_object_management.cpp",
    ].iter().map(|f| sdk.join("src/shared").join(f)).collect();

    let mut all_src = shared_src;
    all_src.push(fi_cpp_patched);

    let mut build = cc::Build::new();
    for i in &includes {
        build.include(i);
    }

    // Add Vulkan SDK include if available.
    if let Ok(vk_sdk) = std::env::var("VULKAN_SDK") {
        build.include(PathBuf::from(vk_sdk).join("include"));
    }

    build
        .files(all_src.iter())
        .cpp(true)
        .std("c++17")
        .define("FFX_GCC", None)
        .define("FFX_FI", None)
        .define("FFX_API_BACKEND", "VK_X64")
        .define("DYNAMIC_LINK_VULKAN", "1")
        .compile("ffx_frameinterpolation");

    println!("cargo:rustc-link-lib=static=ffx_frameinterpolation");
    println!("cargo:rerun-if-changed={}", fi_cpp.display());
}

use std::path::PathBuf;

fn find_sdk() -> PathBuf {
    if let Ok(d) = std::env::var("FIDELITYFX_SDK_DIR") {
        let p = PathBuf::from(d);
        if p.join("sdk/CMakeLists.txt").exists() { return p.join("sdk"); }
        if p.join("CMakeLists.txt").exists() { return p; }
        panic!("bad FIDELITYFX_SDK_DIR");
    }
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    for c in &[root.join("../../tmp/FidelityFX-SDK/sdk"), root.join("../FidelityFX-SDK/sdk"), root.join("FidelityFX-SDK/sdk")] {
        if c.join("CMakeLists.txt").exists() { return c.clone(); }
    }
    panic!("Clone: git clone --depth 1 --branch v1.1.4 https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK.git {}/FidelityFX-SDK", root.display());
}

fn main() {
    if std::env::var("CARGO_FEATURE_FSR3").is_err() { return; }
    let sdk = find_sdk();
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    // Patch ffx_types.h
    let types_out = out.join("include/FidelityFX/host/ffx_types.h");
    if !types_out.exists() {
        std::fs::create_dir_all(types_out.parent().unwrap()).unwrap();
        let s = std::fs::read_to_string(sdk.join("include/FidelityFX/host/ffx_types.h")).unwrap();
        std::fs::write(&types_out, s
            .replace("#pragma warning(", "// #pragma warning(")
            .replace("#define FFX_SDK_DEFAULT_CONTEXT_SIZE (1024 * 128)", "#define FFX_SDK_DEFAULT_CONTEXT_SIZE (1024 * 256)")
            .replace("#define FFX_API __declspec(dllexport)", "#define FFX_API __attribute__((visibility(\"default\")))")
        ).unwrap();
    }

    // Patch frame interpolation source
    let fi_cpp = sdk.join("src/components/frameinterpolation/ffx_frameinterpolation.cpp");
    let fi_out = out.join("ffx_fi.cpp");
    if !fi_out.exists() {
        let s = std::fs::read_to_string(&fi_cpp).unwrap();
        std::fs::write(&fi_out, format!("#define FFX_UNUSED(x) ((void)(x))\n#define _countof(x) (sizeof(x)/sizeof((x)[0]))\n{}",
            s.replace("wcscpy_s", "wcscpy").replace("#include <cfloat>", "#include <cfloat>\n#include <cstring>\n#include <algorithm>"))).unwrap();
    }

    // Patch shared sources
    for f in &["ffx_message.cpp", "ffx_assert.cpp", "ffx_breadcrumbs_list.cpp"] {
        let dst = out.join(f);
        if !dst.exists() {
            let s = std::fs::read_to_string(sdk.join("src/shared").join(f)).unwrap();
            std::fs::write(&dst, format!("#define FFX_UNUSED(x) ((void)(x))\n{}", s)).unwrap();
        }
    }

    let mut b = cc::Build::new();
    b.include(out.join("include"))
     .include(sdk.join("include"))
     .include(sdk.join("src/shared"))
     .include(sdk.join("include/FidelityFX/host"))
     .include(sdk.join("src/components"))
     .include(sdk.join("src/components/frameinterpolation"))
     .include(sdk.join("src/backends/vk"))
     .include(sdk.join("src/backends/shared"))
     .cpp(true).std("c++17")
     .define("FFX_GCC", None)
     .define("FFX_FI", None)
     .define("DYNAMIC_LINK_VULKAN", "1");

    b.file(out.join("ffx_message.cpp"));
    b.file(out.join("ffx_assert.cpp"));
    b.file(out.join("ffx_breadcrumbs_list.cpp"));
    b.file(sdk.join("src/shared/ffx_object_management.cpp"));
    b.file(std::path::PathBuf::from("tmp/ffx_vk_stubs.cpp"));
    b.file(&fi_out);
    b.compile("ffx_frameinterpolation");

    println!("cargo:rustc-link-lib=static=ffx_frameinterpolation");
    println!("cargo:rerun-if-changed={}", fi_cpp.display());
}

#!/bin/sh
# Compile GLSL compute shaders to SPIR-V.
# Requires: glslangValidator (from vulkan-tools or glslang package)
set -e
cd "$(dirname "$0")"
glslangValidator -V motion_est.comp -o motion_est.spv
glslangValidator -V warp_blend.comp -o warp_blend.spv
glslangValidator -V flow_convert.comp -o flow_convert.spv
glslangValidator -V easu_upscale.comp -o easu_upscale.spv
glslangValidator -V rcas.comp -o rcas.spv
echo "Done: motion_est.spv, warp_blend.spv, flow_convert.spv, easu_upscale.spv, rcas.spv"

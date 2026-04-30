#!/bin/sh
# Compile GLSL compute shaders to SPIR-V.
# Requires: glslangValidator (from vulkan-tools or glslang package)
set -e
cd "$(dirname "$0")"
glslangValidator -V motion_est.comp -o motion_est.spv
glslangValidator -V warp_blend.comp -o warp_blend.spv
glslangValidator -V flow_convert.comp -o flow_convert.spv
echo "Done: motion_est.spv, warp_blend.spv, flow_convert.spv"

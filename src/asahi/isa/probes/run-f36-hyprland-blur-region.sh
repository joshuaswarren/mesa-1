#!/bin/sh
set -eu

probe_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$probe_dir/../../../.." && pwd)
build_dir=$(CDPATH= cd -- "${1:-$repo_dir/_build-full}" && pwd)
vulkan_dir=$build_dir/src/asahi/vulkan
source_c=$repo_dir/src/asahi/vulkan/tests/g17p_vulkan_hypr_blur_pingpong.c
icd=$vulkan_dir/asahi_devenv_icd.aarch64.json
run_hardware=${2:-}

cc -std=c11 -O2 -Wall -Wextra -Werror -DF36_REGION=1 \
   '-DTEST_NAME="f36-hyprland-blur-region"' \
   -I"$build_dir/include" -I"$repo_dir/include" \
   "$source_c" -lvulkan -o "$vulkan_dir/f36-hyprland-blur-region"

for pass in 1 2; do
   glslangValidator -V -DUV_SCALE=1.0 \
      "$probe_dir/f35_hyprland_blur_pingpong.vert" \
      -o "$vulkan_dir/f36-hyprland-blur${pass}.vert.spv" >/dev/null
   glslangValidator -V -DFIXED_TEXTURE_SIZE=32.0 \
      "$probe_dir/f34_hyprland_blur${pass}.frag" \
      -o "$vulkan_dir/f36-hyprland-blur${pass}.frag.spv" >/dev/null
   spirv-val "$vulkan_dir/f36-hyprland-blur${pass}.vert.spv"
   spirv-val "$vulkan_dir/f36-hyprland-blur${pass}.frag.spv"
done

echo "f36 Hyprland region: PASS — binary/SPIR-V bundle built in $vulkan_dir"
if [ "$run_hardware" = --hardware ]; then
   export VK_DRIVER_FILES=$icd
   "$vulkan_dir/f36-hyprland-blur-region" \
      "$vulkan_dir/f36-hyprland-blur1.vert.spv" \
      "$vulkan_dir/f36-hyprland-blur1.frag.spv" down
   "$vulkan_dir/f36-hyprland-blur-region" \
      "$vulkan_dir/f36-hyprland-blur2.vert.spv" \
      "$vulkan_dir/f36-hyprland-blur2.frag.spv" up
elif [ -n "$run_hardware" ]; then
   echo "unknown option: $run_hardware (expected --hardware)" >&2
   exit 2
else
   echo "hardware region execution skipped; append --hardware on G17P"
fi

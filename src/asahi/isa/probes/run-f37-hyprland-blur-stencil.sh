#!/bin/sh
set -eu

probe_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$probe_dir/../../../.." && pwd)
build_dir=$(CDPATH= cd -- "${1:-$repo_dir/_build-full}" && pwd)
vulkan_dir=$build_dir/src/asahi/vulkan
source_c=$repo_dir/src/asahi/vulkan/tests/g17p_vulkan_hypr_blur_stencil.c
icd=$vulkan_dir/asahi_devenv_icd.aarch64.json
run_hardware=${2:-}

cc -std=c11 -O2 -Wall -Wextra -Werror \
   -I"$build_dir/include" -I"$repo_dir/include" \
   "$source_c" -lvulkan -o "$vulkan_dir/f37-hyprland-blur-stencil"

glslangValidator -V "$probe_dir/f35_hyprland_blur_pingpong.vert" \
   -o "$vulkan_dir/f37-hyprland-blur-stencil.vert.spv" >/dev/null
for shader in stencil_mask blurred_bg client_composite; do
   glslangValidator -V "$probe_dir/f37_hyprland_${shader}.frag" \
      -o "$vulkan_dir/f37-hyprland-${shader}.frag.spv" >/dev/null
   spirv-val "$vulkan_dir/f37-hyprland-${shader}.frag.spv"
done
spirv-val "$vulkan_dir/f37-hyprland-blur-stencil.vert.spv"

echo "f37 Hyprland stencil: PASS — binary/SPIR-V bundle built in $vulkan_dir"
if [ "$run_hardware" = --hardware ]; then
   export VK_DRIVER_FILES=$icd
   "$vulkan_dir/f37-hyprland-blur-stencil" \
      "$vulkan_dir/f37-hyprland-blur-stencil.vert.spv" \
      "$vulkan_dir/f37-hyprland-stencil_mask.frag.spv" \
      "$vulkan_dir/f37-hyprland-blurred_bg.frag.spv" \
      "$vulkan_dir/f37-hyprland-client_composite.frag.spv"
elif [ -n "$run_hardware" ]; then
   echo "unknown option: $run_hardware (expected --hardware)" >&2
   exit 2
else
   echo "hardware stencil execution skipped; append --hardware on G17P"
fi

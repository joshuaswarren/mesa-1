#!/bin/sh
set -eu

probe_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$probe_dir/../../../.." && pwd)
build_dir=$(CDPATH= cd -- "${1:-$repo_dir/_build-full}" && pwd)
vulkan_dir=$build_dir/src/asahi/vulkan
source_c=$repo_dir/src/asahi/vulkan/tests/g17p_vulkan_hypr_blur_pingpong.c
graphics_probe=$vulkan_dir/g17p-vulkan-graphics-probe
shim=$build_dir/src/asahi/drm-shim/libasahi_noop_drm_shim.so
icd=$vulkan_dir/asahi_devenv_icd.aarch64.json
run_hardware=${2:-}

for file in "$source_c" "$graphics_probe" "$shim" "$icd" \
            "$probe_dir/f35_hyprland_blur_pingpong.vert" \
            "$probe_dir/f34_hyprland_blur1.frag" \
            "$probe_dir/f34_hyprland_blur2.frag"; do
   test -r "$file" || {
      echo "missing probe dependency: $file" >&2
      exit 2
   }
done

cc -std=c11 -O2 -Wall -Wextra -Werror \
   -I"$build_dir/include" -I"$repo_dir/include" \
   "$source_c" -lvulkan -o "$vulkan_dir/f35-hyprland-blur-pingpong"

for pass in 1 2; do
   scale=$([ "$pass" = 1 ] && echo 0.5 || echo 2.0)
   glslangValidator -V -DUV_SCALE="$scale" \
      "$probe_dir/f35_hyprland_blur_pingpong.vert" \
      -o "$vulkan_dir/f35-hyprland-blur${pass}.vert.spv" >/dev/null
   glslangValidator -V -DFIXED_TEXTURE_SIZE=32.0 \
      "$probe_dir/f34_hyprland_blur${pass}.frag" \
      -o "$vulkan_dir/f35-hyprland-blur${pass}.frag.spv" >/dev/null
   spirv-val "$vulkan_dir/f35-hyprland-blur${pass}.vert.spv"
   spirv-val "$vulkan_dir/f35-hyprland-blur${pass}.frag.spv"

   # Compile a binding-1 copy under drm-shim so pipeline encodability remains
   # host-testable even though the stored-attachment result requires hardware.
   temp_frag=$(mktemp)
   trap 'rm -f "$temp_frag"' EXIT HUP INT TERM
   glslangValidator -V -DTEX_BINDING=1 -DFIXED_TEXTURE_SIZE=32.0 \
      "$probe_dir/f34_hyprland_blur${pass}.frag" -o "$temp_frag" >/dev/null
   LD_PRELOAD=$shim \
   ASAHI_SHIM_CHIP=g17p \
   VK_DRIVER_FILES=$icd \
   MESA_SHADER_CACHE_DISABLE=true \
   DEPTH_FORMAT=0 \
   COLOR_FORMAT=37 \
      "$graphics_probe" "$vulkan_dir/f35-hyprland-blur${pass}.vert.spv" \
      "$temp_frag" | grep -Fq 'ENCODES (VkResult 0)'
   rm -f "$temp_frag"
   trap - EXIT HUP INT TERM
done

echo "f35 Hyprland ping-pong: PASS — binary/SPIR-V bundle built in $vulkan_dir"

if [ "$run_hardware" = --hardware ]; then
   export VK_DRIVER_FILES=$icd
   "$vulkan_dir/f35-hyprland-blur-pingpong" \
      "$vulkan_dir/f35-hyprland-blur1.vert.spv" \
      "$vulkan_dir/f35-hyprland-blur1.frag.spv"
   "$vulkan_dir/f35-hyprland-blur-pingpong" \
      "$vulkan_dir/f35-hyprland-blur2.vert.spv" \
      "$vulkan_dir/f35-hyprland-blur2.frag.spv"
elif [ -n "$run_hardware" ]; then
   echo "unknown option: $run_hardware (expected --hardware)" >&2
   exit 2
else
   echo "hardware ping-pong execution skipped; append --hardware on G17P"
fi

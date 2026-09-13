#!/bin/sh
set -eu

probe_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$probe_dir/../../../.." && pwd)
build_dir=$(CDPATH= cd -- "${1:-$repo_dir/_build-full}" && pwd)
vulkan_dir=$build_dir/src/asahi/vulkan
graphics_probe=$vulkan_dir/g17p-vulkan-graphics-probe
shim=$build_dir/src/asahi/drm-shim/libasahi_noop_drm_shim.so
icd=$vulkan_dir/asahi_devenv_icd.aarch64.json
harness=$repo_dir/src/asahi/vulkan/tests/g17p_offscreen_triangle.c
run_hardware=${2:-}

for file in "$graphics_probe" "$shim" "$icd" "$harness" \
            "$probe_dir/f34_hyprland_blur1.vert" \
            "$probe_dir/f34_hyprland_blur1.frag" \
            "$probe_dir/f34_hyprland_blur2.vert" \
            "$probe_dir/f34_hyprland_blur2.frag"; do
   test -r "$file" || {
      echo "missing probe dependency: $file" >&2
      exit 2
   }
done

temp_dir=$(mktemp -d)
cleanup()
{
   rm -f "$temp_dir"/*
   rmdir "$temp_dir"
}
trap cleanup EXIT HUP INT TERM

compile_shader()
{
   stage=$1
   source=$2
   output=$3
   shift 3
   glslangValidator -V "$@" -S "$stage" "$source" -o "$output" >/dev/null
   spirv-val "$output"
}

make_header()
{
   input=$1
   symbol=$2
   output=$3
   {
      echo '#include <stdint.h>'
      echo "const uint32_t ${symbol}[] = {"
      od -An -v -t x4 "$input" |
         awk '{ for (i = 1; i <= NF; ++i) printf "0x%s,", $i; print "" }'
      echo '};'
   } >"$output"
}

for pass in 1 2; do
   compile_shader vert "$probe_dir/f34_hyprland_blur${pass}.vert" \
      "$temp_dir/blur${pass}.vert.spv" -DTEX_BINDING=1
   compile_shader frag "$probe_dir/f34_hyprland_blur${pass}.frag" \
      "$temp_dir/blur${pass}.frag.spv" -DTEX_BINDING=1

   LD_PRELOAD=$shim \
   ASAHI_SHIM_CHIP=g17p \
   VK_DRIVER_FILES=$icd \
   AGX_MESA_DEBUG=shaders \
   MESA_SHADER_CACHE_DISABLE=true \
   DEPTH_FORMAT=0 \
   COLOR_FORMAT=37 \
      "$graphics_probe" "$temp_dir/blur${pass}.vert.spv" \
      "$temp_dir/blur${pass}.frag.spv" >"$temp_dir/blur${pass}.log" 2>&1

   grep -Fq 'ENCODES (VkResult 0)' "$temp_dir/blur${pass}.log"
   samples=$(grep -Ec '32x4 +%[0-9]+ = \(float32\)tex ' \
      "$temp_dir/blur${pass}.log")
   expected=$([ "$pass" = 1 ] && echo 5 || echo 8)
   test "$samples" -eq "$expected" || {
      echo "blur${pass}: expected $expected texture operations, got $samples" >&2
      exit 1
   }
   echo "f34 Hyprland blur${pass}: ENCODES with $samples implicit texture operations"

   # Rebind the exact shader pair into the existing, independently verified
   # implicit-LOD pixel harness. The harness supplies solid red/green/blue
   # mip levels, LINEAR filtering, an RGBA8 target, blending disabled, and an
   # exact green readback oracle. Weighted sums of the selected green mip must
   # remain exactly green for both kernels.
   compile_shader vert "$probe_dir/f34_hyprland_blur${pass}.vert" \
      "$temp_dir/pixel${pass}.vert.spv"
   compile_shader frag "$probe_dir/f34_hyprland_blur${pass}.frag" \
      "$temp_dir/pixel${pass}.frag.spv"
   make_header "$temp_dir/pixel${pass}.vert.spv" \
      "f34_hyprland_blur${pass}_vert_spv" \
      "$temp_dir/f34_hyprland_blur${pass}_vert.h"
   make_header "$temp_dir/pixel${pass}.frag.spv" \
      "f34_hyprland_blur${pass}_frag_spv" \
      "$temp_dir/f34_hyprland_blur${pass}_frag.h"
   sed \
      -e "s#g17p_vulkan_implicit_lod_vert#f34_hyprland_blur${pass}_vert#g" \
      -e "s#g17p_vulkan_implicit_lod_frag#f34_hyprland_blur${pass}_frag#g" \
      -e "s#g17p-vulkan-implicit-lod#f34-hyprland-blur${pass}#g" \
      "$harness" >"$temp_dir/blur${pass}.c"
   cc -std=c11 -O2 -D_GNU_SOURCE -DG17P_IMPLICIT_LOD=1 \
      -I"$temp_dir" -I"$build_dir/include" -I"$repo_dir/include" \
      "$temp_dir/blur${pass}.c" -o "$temp_dir/f34-hyprland-blur${pass}"
   install -m 755 "$temp_dir/f34-hyprland-blur${pass}" \
      "$vulkan_dir/f34-hyprland-blur${pass}"
done

echo "f34 Hyprland blur: PASS — exact current shader shapes compile and pixel oracles are in $vulkan_dir"

if [ "$run_hardware" = --hardware ]; then
   unset LD_PRELOAD ASAHI_SHIM_CHIP AGX_MESA_DEBUG MESA_SHADER_CACHE_DISABLE
   export VK_DRIVER_FILES=$icd
   "$vulkan_dir/f34-hyprland-blur1"
   "$vulkan_dir/f34-hyprland-blur2"
elif [ -n "$run_hardware" ]; then
   echo "unknown option: $run_hardware (expected --hardware)" >&2
   exit 2
else
   echo "hardware pixel execution skipped; append --hardware on the G17P machine"
fi

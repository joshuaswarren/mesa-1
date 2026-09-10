#!/bin/sh
set -eu

probe_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
build_dir=$(CDPATH= cd -- "${1:-$probe_dir/../../../../_build-full}" && pwd)
vulkan_dir=$build_dir/src/asahi/vulkan
graphics_probe=$vulkan_dir/g17p-vulkan-graphics-probe
shim=$build_dir/src/asahi/drm-shim/libasahi_noop_drm_shim.so
icd=$vulkan_dir/asahi_devenv_icd.aarch64.json

for file in "$graphics_probe" "$shim" "$icd" \
            "$probe_dir/common.vert" \
            "$probe_dir/f33_quickshell_abs_scale.frag"; do
   test -r "$file" || {
      echo "missing probe dependency: $file" >&2
      exit 2
   }
done

temp_dir=$(mktemp -d)
cleanup()
{
   rm -f "$temp_dir/common.vert.spv" "$temp_dir/f33.frag.spv" \
         "$temp_dir/probe.log"
   rmdir "$temp_dir"
}
trap cleanup EXIT HUP INT TERM

glslangValidator -V "$probe_dir/common.vert" \
   -o "$temp_dir/common.vert.spv" >/dev/null
glslangValidator -V "$probe_dir/f33_quickshell_abs_scale.frag" \
   -o "$temp_dir/f33.frag.spv" >/dev/null
spirv-val "$temp_dir/common.vert.spv"
spirv-val "$temp_dir/f33.frag.spv"

export LD_PRELOAD=$shim
export ASAHI_SHIM_CHIP=g17p
export VK_DRIVER_FILES=$icd
export AGX_MESA_DEBUG=shaders
export MESA_SHADER_CACHE_DISABLE=true

"$graphics_probe" "$temp_dir/common.vert.spv" \
   "$temp_dir/f33.frag.spv" 2>&1 | tee "$temp_dir/probe.log"

grep -Eq 'fmul .*\.abs, #2\.000000' "$temp_dir/probe.log"
grep -Fq 'ENCODES (VkResult 0)' "$temp_dir/probe.log"
if grep -Eq 'no grounded AGX3 encoding|pipeline creation failed' \
      "$temp_dir/probe.log"; then
   echo "f33 QuickShell abs-scale unexpectedly refused" >&2
   exit 1
fi

echo "f33 QuickShell abs-scale: PASS — exact former refusal reaches the grounded G17P encoder"

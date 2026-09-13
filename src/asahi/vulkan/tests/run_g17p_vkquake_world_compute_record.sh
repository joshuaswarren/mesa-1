#!/bin/sh
# Offline arm: qualifies pipeline creation and the record-time admission
# gates under the DRM shim.  For the hardware arm run the three binaries
# directly, with no shim and no G17P_VKQUAKE_COMPUTE_RECORD_ONLY: they then
# submit and compare the generated indirect commands, the generated index
# buffer and every lightmap texel against exact expected values.
set -eu

package_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

for file in world-gen lightmap world-compute; do
   test -x "$package_dir/g17p-vulkan-vkquake-$file" || {
      echo "missing oracle: g17p-vulkan-vkquake-$file" >&2
      exit 2
   }
done

for file in libvulkan_asahi.so libasahi_noop_drm_shim.so icd.json; do
   test -r "$package_dir/$file" || {
      echo "missing package file: $file" >&2
      exit 2
   }
done

# ASAHI_SHIM_CHIP must be exactly g17p: any other value silently falls back to
# AGX2, which would make a G17P-only refusal look like a pass.
export ASAHI_SHIM_CHIP=g17p
export ASAHI_SHIM_VALIDATE_G17P_STREAM=1
export G17P_VKQUAKE_COMPUTE_RECORD_ONLY=1
export LD_PRELOAD="$package_dir/libasahi_noop_drm_shim.so"
export MESA_SHADER_CACHE_DISABLE=true
export VK_DRIVER_FILES="$package_dir/icd.json"

# HK_DEBUG_RECORD names any refused recording call with its file and line.
# Without it a refused vkCmd* is silent, the command buffer goes INVALID, and
# the work it carries disappears with no diagnostic anywhere.
export HK_DEBUG_RECORD=1

for mode in world-gen lightmap world-compute; do
   "$package_dir/g17p-vulkan-vkquake-$mode"
done

echo "g17p-vulkan-vkquake-world-compute: PASS — vkQuake's brush-culling and lightmap kernels compile, their static texture uploads record, and both dispatches record"

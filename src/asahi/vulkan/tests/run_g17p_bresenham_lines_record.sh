#!/bin/sh
set -eu

package_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
log=$(mktemp)
trap 'rm -f "$log"' EXIT HUP INT TERM

for file in g17p-vulkan-bresenham-lines libvulkan_asahi.so \
            libasahi_noop_drm_shim.so g17p_wsi_triangle_icd.json; do
   test -r "$package_dir/$file" || {
      echo "missing package file: $package_dir/$file" >&2
      exit 2
   }
done

export ASAHI_SHIM_CHIP=g17p
export G17P_BRESENHAM_LINES_RECORD_ONLY=1
export HK_G17P_LAUNCH_TRACE=1
export LD_PRELOAD="$package_dir/libasahi_noop_drm_shim.so"

"$package_dir/run_g17p_bresenham_lines.sh" 2>&1 | tee "$log"

# Exact little-endian records inside the public Vulkan command buffer:
# PRIMARY_FACE has width 0x2f (3px) on both faces; SECONDARY_FACE has native
# object type LINE (1) on both faces; CULL has Primitive MSAA bit15 set.
grep -Fq 'b400000100020400002f20070000000e002f20070000000e00000000' "$log"
grep -Fq '48000002000200000000e0170000e017ffff0100' "$log"
grep -Fq '0000200080840100' "$log"

echo "g17p-vulkan-bresenham-lines-record: PASS — exact PRIMARY, SECONDARY, CULL and vkEndCommandBuffer"

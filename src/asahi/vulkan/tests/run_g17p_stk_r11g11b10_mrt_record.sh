#!/bin/sh
set -eu

package_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
log=$(mktemp)
trap 'rm -f "$log"' EXIT HUP INT TERM

for file in g17p-vulkan-stk-r11g11b10-mrt libvulkan_asahi.so \
            libasahi_noop_drm_shim.so icd.json; do
   test -r "$package_dir/$file" || {
      echo "missing package file: $file" >&2
      exit 2
   }
done

export ASAHI_SHIM_CHIP=g17p
export ASAHI_SHIM_VALIDATE_G17P_STREAM=1
export G17P_RGBA8_MRT_RECORD_ONLY=1
export LD_PRELOAD="$package_dir/libasahi_noop_drm_shim.so"
export MESA_SHADER_CACHE_DISABLE=true
export VK_DRIVER_FILES="$package_dir/icd.json"

if "$package_dir/g17p-vulkan-stk-r11g11b10-mrt" >"$log" 2>&1; then
   cat "$log"
   echo "unexpectedly admitted unsupported dual R11G11B10 MRT" >&2
   exit 1
fi
cat "$log"
grep -Fq 'vkEndCommandBuffer returned -8' "$log"

echo "g17p-vulkan-stk-r11g11b10-mrt-record: PASS — exact STK MRT reproduces the current fail-closed admission gap"

#!/bin/sh
set -eu

package_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

export VK_DRIVER_FILES="$package_dir/g17p_wsi_triangle_icd.json"
export LD_LIBRARY_PATH="$package_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# G17P_A2R10G10B10_UNORM_RECORD_ONLY=1 is intended for DRM-shim: it still
# requires format advertisement, pipeline compilation and successful EndCB.
# With no override, the binary checks pixels, guards and stability.
exec "$package_dir/g17p-vulkan-a2r10g10b10-unorm"

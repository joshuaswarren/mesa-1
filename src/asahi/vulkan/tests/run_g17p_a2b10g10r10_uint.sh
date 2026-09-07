#!/bin/sh
set -eu

package_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

export VK_DRIVER_FILES="$package_dir/g17p_wsi_triangle_icd.json"
export LD_LIBRARY_PATH="$package_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# G17P_A2B10G10R10_UINT_RECORD_ONLY=1 is intended for DRM-shim: it still
# requires format advertisement, pipeline compilation and successful EndCB.
# With no override, all pixels, guards and 3s stability are checked.
exec "$package_dir/g17p-vulkan-a2b10g10r10-uint"

#!/usr/bin/env bash
# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT
# Binaries and shaders must be built elsewhere before running on the Neo.
# Invoke inside a bounded GPU-session guard through aurora-ctl hw request.
set -euo pipefail
prefix=${1:?Mesa prefix}
probes=${2:?directory containing the four prebuilt probes}
export VK_DRIVER_FILES="$prefix/share/vulkan/icd.d/asahi_icd.aarch64.json"
export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
export MESA_SHADER_CACHE_DISABLE=true
# Enable the real (conformant, slower) implementation also used by vkd3d's
# application profile. Both reference and candidate use this same option.
export hk_image_view_min_lod=true
for op in tex txb txl txd; do
   for clamp in 0 2; do
      printf 'op=%s image-view minLod=%s\n' "$op" "$clamp"
      G17P_VIEW_MIN_LOD="$clamp" timeout -k 2 40 \
         "$probes/g17p-vulkan-min-lod-$op"
   done
done

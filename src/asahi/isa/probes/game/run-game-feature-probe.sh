#!/bin/sh
# SPDX-License-Identifier: MIT
#
# Offline G17P game-feature matrix.  Needs no hardware: it runs the Honeykrisp
# ICD against the G17P drm-shim, enables every advertised Vulkan 1.1/1.2/1.3
# feature the way DXVK and VKD3D-Proton do, and records the compile and record
# result of each probe shader.
#
# The output is compared against expected-g17p.txt, which is the measured
# matrix at the time it was checked in -- NOT a statement that every ok row is
# hardware-qualified.  A diff means the advertising-versus-reality matrix
# moved, in either direction, and the baseline needs re-reading before it is
# updated.
#
#   usage: run-game-feature-probe.sh <meson build dir> [--update]
set -eu

build=${1:?"usage: run-game-feature-probe.sh <meson build dir> [--update]"}
mode=${2:-check}
here=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)

probe="$build/src/asahi/vulkan/g17p-vulkan-feature-probe"
shim="$build/src/asahi/drm-shim/libasahi_noop_drm_shim.so"
icd="$build/src/asahi/vulkan/asahi_devenv_icd.aarch64.json"

for f in "$probe" "$shim" "$icd"; do
   if [ ! -r "$f" ]; then
      echo "missing build artifact: $f" >&2
      echo "build it with: ninja -C $build \\" >&2
      echo "  src/asahi/vulkan/g17p-vulkan-feature-probe \\" >&2
      echo "  src/asahi/drm-shim/libasahi_noop_drm_shim.so \\" >&2
      echo "  src/asahi/vulkan/libvulkan_asahi.so" >&2
      exit 2
   fi
done

# The probe reports each module twice: once with update-after-bind enabled,
# which is what a translation layer does, and once without.  Several G17P
# descriptor lowerings in hk_compile_nir() are gated on the sampled-image
# update-after-bind feature being OFF, so the two runs are not redundant.
run_probe() {
   ASAHI_SHIM_CHIP=g17p \
   LD_PRELOAD="$shim" \
   VK_DRIVER_FILES="$icd" \
   MESA_SHADER_CACHE_DISABLE=true \
      "$probe" "$@" 2>/dev/null | grep -E '^#|COMPILE=' || true
}

{
   echo "=== update-after-bind enabled (translation-layer default) ==="
   run_probe "$here"/*.spv
   echo "=== update-after-bind disabled ==="
   run_probe --no-update-after-bind "$here"/*.spv
} > "$build/g17p-game-features.txt"

if [ "$mode" = "--update" ]; then
   cp "$build/g17p-game-features.txt" "$here/expected-g17p.txt"
   echo "baseline updated: $here/expected-g17p.txt"
   exit 0
fi

if diff -u "$here/expected-g17p.txt" "$build/g17p-game-features.txt"; then
   echo "G17P game-feature matrix matches the checked-in baseline"
else
   echo "G17P game-feature matrix MOVED -- read the diff before updating" >&2
   exit 1
fi

#!/bin/sh
set -eu

# Builds and gates the Qt 6 HiQ subpixel distance-field text oracle.
#
# Without --hardware this only proves the bundle builds, the SPIR-V validates,
# and the G17P driver encodes every variant under drm-shim with stream
# validation on.  Nothing executes under the shim, so no pixel claim can be
# made there.
#
# With --hardware on Neo it runs a ladder in which each rung moves exactly one
# variable away from the previous one:
#
#   center    .a through a zero/zero/zero/red swizzle, no blend   must PASS
#   center-r  the same texel through .r and an identity swizzle   must PASS
#   blend     the center shader plus Qt's blend                   must PASS
#   taps      four offset implicit-LOD taps, no blend             must PASS
#   broken    the same taps with the projector divide dropped     must FAIL
#   hiq       the whole material                                  must PASS
#   negative  hiq against a reference shifted one texel           must FAIL
#
# The two must-FAIL rungs are what make the must-PASS rungs mean anything: a
# probe that compiled nothing, or a comparison that accepts any image, reports
# success on the PASS rungs too.
#
# Deliberately absent: a textureProj-versus-manual-divide A/B.  It has no
# discriminating power.  nir_lower_tex sets lower_txp = ~0 for AGX
# (agx_nir_lower_texture.c:801), and dumping both with AGX_MESA_DEBUG=shaders
# gives byte-identical AGX IR apart from the source hash.

probe_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$probe_dir/../../../.." && pwd)
build_dir=$(CDPATH= cd -- "${1:-$repo_dir/_build-full}" && pwd)
vulkan_dir=$build_dir/src/asahi/vulkan
source_c=$repo_dir/src/asahi/vulkan/tests/g17p_vulkan_qt_subpixel_text.c
graphics_probe=$vulkan_dir/g17p-vulkan-graphics-probe
shim=$build_dir/src/asahi/drm-shim/libasahi_noop_drm_shim.so
icd=$vulkan_dir/asahi_devenv_icd.aarch64.json
binary=$vulkan_dir/f41-qt-subpixel-text
run_hardware=${2:-}

for file in "$source_c" "$graphics_probe" "$shim" "$icd" \
            "$probe_dir/f41_qt_subpixel_text.vert" \
            "$probe_dir/f41_qt_subpixel_text.frag"; do
   test -r "$file" || {
      echo "missing probe dependency: $file" >&2
      exit 2
   }
done

cc -std=c11 -O2 -Wall -Wextra -Werror \
   -I"$build_dir/include" -I"$repo_dir/include" \
   "$source_c" -lvulkan -lm -o "$binary"

glslangValidator -V "$probe_dir/f41_qt_subpixel_text.vert" \
   -o "$vulkan_dir/f41-qt-subpixel-text.vert.spv" >/dev/null
spirv-val "$vulkan_dir/f41-qt-subpixel-text.vert.spv"

build_frag() {
   name=$1
   shift
   glslangValidator -V "$@" "$probe_dir/f41_qt_subpixel_text.frag" \
      -o "$vulkan_dir/f41-qt-subpixel-text-$name.frag.spv" >/dev/null
   spirv-val "$vulkan_dir/f41-qt-subpixel-text-$name.frag.spv"

   LD_PRELOAD=$shim \
   ASAHI_SHIM_CHIP=g17p \
   ASAHI_SHIM_VALIDATE_G17P_STREAM=1 \
   VK_DRIVER_FILES=$icd \
   MESA_SHADER_CACHE_DISABLE=true \
   DEPTH_FORMAT=0 \
   COLOR_FORMAT=37 \
      "$graphics_probe" "$vulkan_dir/f41-qt-subpixel-text.vert.spv" \
      "$vulkan_dir/f41-qt-subpixel-text-$name.frag.spv" |
      grep -Fq 'ENCODES (VkResult 0)' || {
      echo "f41 $name: G17P refused to encode the pipeline" >&2
      exit 1
   }
   echo "f41 $name: encodes on G17P under drm-shim"
}

build_frag center -DPROBE_CENTER
build_frag center-r -DPROBE_CENTER -DREAD_R
build_frag taps -DPROBE_TAPS
build_frag broken -DPROBE_TAPS -DBREAK_PROJ
build_frag hiq

echo "f41 Qt subpixel distance-field text: PASS — bundle built in $vulkan_dir"

if [ "$run_hardware" != --hardware ]; then
   if [ -n "$run_hardware" ]; then
      echo "unknown option: $run_hardware (expected --hardware)" >&2
      exit 2
   fi
   echo "hardware execution skipped; append --hardware on G17P"
   exit 0
fi

export VK_DRIVER_FILES=$icd
vert=$vulkan_dir/f41-qt-subpixel-text.vert.spv

expect_pass() {
   name=$1
   mode=$2
   shift 2
   echo "== f41 $name ($mode): expecting PASS"
   "$binary" "$vert" "$vulkan_dir/f41-qt-subpixel-text-$name.frag.spv" \
      "$mode" "$@"
}

expect_fail() {
   name=$1
   mode=$2
   shift 2
   echo "== f41 $name ($mode): expecting FAIL"
   if "$binary" "$vert" "$vulkan_dir/f41-qt-subpixel-text-$name.frag.spv" \
         "$mode" "$@"; then
      echo "f41 $name: control passed when it had to fail; the oracle is not" \
           "discriminating and its PASS results mean nothing" >&2
      exit 1
   fi
   echo "f41 $name: failed as required"
}

expect_pass center center
expect_pass center-r center --identity-swizzle
expect_pass center blend
expect_pass taps taps
expect_fail broken taps
expect_pass hiq hiq
expect_fail hiq hiq --negative

echo "f41 Qt subpixel distance-field text: hardware ladder complete"

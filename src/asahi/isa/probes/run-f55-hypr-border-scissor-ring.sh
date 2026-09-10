#!/bin/sh
set -eu

# Gates the Hyprland border scissor-ring oracle.
#
#   run-f55-hypr-border-scissor-ring.sh [build-dir] [--hardware]
#
# Without --hardware this only proves the G17P driver encodes and records each
# mode under the no-op drm-shim with stream validation on. Nothing executes
# there, so no pixel claim can be made; the encode gate cannot see a scissor
# that clips one row short, because the record is byte-legal either way.
#
# With --hardware on Neo it runs the ladder, each rung moving exactly one
# variable from the previous one:
#
#   whole    one draw, scissor = the whole attachment       must PASS
#   tile     four rectangles tiling the whole attachment    must PASS
#   ring     Hyprland's four-rectangle border ring          must PASS
#   damage   the ring again over a loaded previous frame    must PASS
#   gap      the ring with one rectangle one row short      must FAIL
#   overlap  the ring with one rectangle one row long       must FAIL
#
# The two must-FAIL rungs are what make the must-PASS rungs mean anything: a
# run in which nothing executed, or a comparison that accepts any image,
# reports success on the PASS rungs too.
#
# What each rung decides, if it is the one that fails:
#
#   whole   fails  -> the failure is not about the scissor at all.
#   tile    fails  -> adjacent scissor rectangles do not tile; the seam is
#                     generic, not specific to Hyprland's ring geometry.
#   ring    fails  -> the seam only appears for an off-origin ring, which is
#                     what a window border is and what no witness covers.
#   damage  fails  -> the ring is fine on a cleared attachment and wrong on a
#                     loaded one, so the fault is in partial-render load, not
#                     in the scissor.
#
# Read the per-pixel lines the binary prints: it reports each wrong pixel as
# "covered N time(s), expected M". N < M is a hole, N > M is a doubly blended
# seam, and the coordinates say which rectangle edge it is.

probe_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$probe_dir/../../../.." && pwd)
build_dir=$(CDPATH= cd -- "${1:-$repo_dir/_build}" && pwd)
vulkan_dir=$build_dir/src/asahi/vulkan
binary=$vulkan_dir/g17p-vulkan-scissor-ring
shim=$build_dir/src/asahi/drm-shim/libasahi_noop_drm_shim.so
icd=$vulkan_dir/asahi_devenv_icd.aarch64.json
run_hardware=${2:-}

for file in "$binary" "$shim" "$icd"; do
   test -r "$file" || {
      echo "missing probe dependency: $file" >&2
      exit 2
   }
done

modes='whole tile ring damage gap overlap'

for mode in $modes; do
   LD_PRELOAD=$shim \
   ASAHI_SHIM_CHIP=g17p \
   ASAHI_SHIM_VALIDATE_G17P_STREAM=1 \
   VK_DRIVER_FILES=$icd \
   MESA_SHADER_CACHE_DISABLE=true \
      "$binary" "$mode" --record-only |
      grep -Fq 'RECORDS (VkResult 0)' || {
      echo "f55 $mode: G17P refused to encode or record the mode" >&2
      exit 1
   }
   echo "f55 $mode: encodes and records on G17P under drm-shim"
done

echo "f55 Hyprland border scissor ring: PASS — every mode encodes"

if [ "$run_hardware" != --hardware ]; then
   if [ -n "$run_hardware" ]; then
      echo "unknown option: $run_hardware (expected --hardware)" >&2
      exit 2
   fi
   echo "hardware execution skipped; append --hardware on G17P"
   exit 0
fi

export VK_DRIVER_FILES=$icd

for mode in whole tile ring damage; do
   echo "== f55 $mode: expecting PASS"
   "$binary" "$mode"
done

for mode in gap overlap; do
   echo "== f55 $mode: expecting the comparison to reject it"
   "$binary" "$mode"
done

echo "f55 Hyprland border scissor ring: HARDWARE PASS — adjacent scissor" \
     "rectangles tile without a hole or a doubly blended seam, and both" \
     "one-row controls were rejected"

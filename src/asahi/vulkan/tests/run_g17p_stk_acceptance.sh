#!/bin/sh
# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT

set -eu

BASE_MESA_COMMIT=044cfce974b340fc0d2ea2f4d9574c0a78a32fa6
BASE_KERNEL_COMMIT=3f0148d76
timeout_s=120
visual_diff_ppm=10000
max_cold_ms=30000
max_warm_ms=15000
max_vdm=1000
min_dispatches=5000
stk_bin=supertuxkart
capture_program=grim
compare_program=compare
identify_program=identify
output_dir=
config_source=
menu_reference=
tutorial_reference=
tutorial_hook=
mesa_source=
expected_mesa_commit=
icd_file=
driver_library=
expected_driver_sha256=
kernel_source=
expected_kernel_commit=
expected_kernel_release=

usage()
{
   cat <<EOF
Usage: $0 [options] [-- SuperTuxKart arguments]

Required:
  --output DIR                 New result directory
  --config-source DIR          Known menu-ready XDG_CONFIG_HOME template
  --menu-reference PNG         Known-good menu screenshot
  --tutorial-reference PNG     Known-good tutorial screenshot
  --tutorial-hook PROGRAM      PROGRAM <stk-pid> navigates menu to tutorial
  --mesa-source DIR            Git worktree used for the deployed build
  --mesa-commit SHA            Exact deployed Mesa commit
  --icd FILE                   ICD JSON naming --driver-library exactly
  --driver-library FILE        Deployed libvulkan_asahi.so
  --driver-sha256 SHA256       Exact deployed driver digest
  --kernel-source DIR          Git worktree used for the running kernel
  --kernel-commit SHA          Exact deployed kernel commit
  --kernel-release STRING      Exact expected uname -r

Optional:
  --timeout SECONDS            Per-screen deadline (default: 120)
  --visual-diff-ppm N          Allowed differing pixels per million (10000)
  --max-cold-ms N              Cold menu deadline (30000)
  --max-warm-ms N              Warm menu deadline (15000)
  --max-vdm N                  Maximum render streams per run (1000)
  --min-dispatches N           Minimum CDM launches per run (5000)
  --stk FILE                   SuperTuxKart executable
  --capture-program FILE       FILE <output.png> (default: grim)
  --compare-program FILE       ImageMagick compare-compatible executable

Cold and warm runs share a fresh, isolated shader cache but receive separate
copies of the supplied configuration. The runner enables only the aggregate
HK_G17P_SUBMIT_TRACE diagnostic. It rejects synchronization, stream-splitting,
and per-submit batching controls so the measured path is the production path.
EOF
}

while [ "$#" -gt 0 ]; do
   case "$1" in
   --output) output_dir=$2; shift 2 ;;
   --config-source) config_source=$2; shift 2 ;;
   --menu-reference) menu_reference=$2; shift 2 ;;
   --tutorial-reference) tutorial_reference=$2; shift 2 ;;
   --tutorial-hook) tutorial_hook=$2; shift 2 ;;
   --mesa-source) mesa_source=$2; shift 2 ;;
   --mesa-commit) expected_mesa_commit=$2; shift 2 ;;
   --icd) icd_file=$2; shift 2 ;;
   --driver-library) driver_library=$2; shift 2 ;;
   --driver-sha256) expected_driver_sha256=$2; shift 2 ;;
   --kernel-source) kernel_source=$2; shift 2 ;;
   --kernel-commit) expected_kernel_commit=$2; shift 2 ;;
   --kernel-release) expected_kernel_release=$2; shift 2 ;;
   --timeout) timeout_s=$2; shift 2 ;;
   --visual-diff-ppm) visual_diff_ppm=$2; shift 2 ;;
   --max-cold-ms) max_cold_ms=$2; shift 2 ;;
   --max-warm-ms) max_warm_ms=$2; shift 2 ;;
   --max-vdm) max_vdm=$2; shift 2 ;;
   --min-dispatches) min_dispatches=$2; shift 2 ;;
   --stk) stk_bin=$2; shift 2 ;;
   --capture-program) capture_program=$2; shift 2 ;;
   --compare-program) compare_program=$2; shift 2 ;;
   --) shift; break ;;
   -h|--help) usage; exit 0 ;;
   *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
   esac
done

for value in output_dir config_source menu_reference tutorial_reference \
             tutorial_hook mesa_source expected_mesa_commit driver_library \
             icd_file expected_driver_sha256 kernel_source \
             expected_kernel_commit expected_kernel_release; do
   eval "resolved=\${$value}"
   if [ -z "$resolved" ]; then
      echo "Missing required option for $value" >&2
      usage >&2
      exit 2
   fi
done

case "$output_dir" in
""|/|.|..) echo "Refusing unsafe output directory '$output_dir'" >&2; exit 2 ;;
esac
if [ -e "$output_dir" ]; then
   echo "Output directory already exists: $output_dir" >&2
   exit 2
fi

for file in "$menu_reference" "$tutorial_reference" "$tutorial_hook" \
            "$icd_file" "$driver_library"; do
   if [ ! -f "$file" ]; then
      echo "Missing file: $file" >&2
      exit 2
   fi
done
if [ ! -x "$tutorial_hook" ]; then
   echo "Tutorial hook is not executable: $tutorial_hook" >&2
   exit 2
fi
if [ ! -d "$config_source" ]; then
   echo "Invalid config source directory: $config_source" >&2
   exit 2
fi
if ! git -C "$mesa_source" rev-parse --git-dir >/dev/null 2>&1; then
   echo "Invalid Mesa source directory: $mesa_source" >&2
   exit 2
fi
if ! git -C "$kernel_source" rev-parse --git-dir >/dev/null 2>&1; then
   echo "Invalid kernel source directory: $kernel_source" >&2
   exit 2
fi
for program in "$stk_bin" "$capture_program" "$compare_program" \
               "$identify_program" git sha256sum vulkaninfo journalctl awk; do
   if ! command -v "$program" >/dev/null 2>&1; then
      echo "Required program not found: $program" >&2
      exit 2
   fi
done

for name in ZINK_DEBUG HK_PERFTEST HK_G17P_DRAWS_PER_STREAM \
            HK_G17P_STATE_PUBLICATIONS_PER_STREAM \
            HK_G17P_STRONG_DISPATCH_FLUSH; do
   eval "value=\${$name-}"
   if [ -n "$value" ]; then
      echo "Refusing path-altering environment: $name=$value" >&2
      exit 2
   fi
done

actual_mesa_commit=$(git -C "$mesa_source" rev-parse HEAD)
if [ "$actual_mesa_commit" != "$expected_mesa_commit" ]; then
   echo "Mesa commit mismatch: $actual_mesa_commit != $expected_mesa_commit" >&2
   exit 1
fi
if ! git -C "$mesa_source" merge-base --is-ancestor \
      "$BASE_MESA_COMMIT" "$actual_mesa_commit"; then
   echo "Mesa build does not contain the G17P mip fix $BASE_MESA_COMMIT" >&2
   exit 1
fi
actual_kernel_commit=$(git -C "$kernel_source" rev-parse HEAD)
if [ "$actual_kernel_commit" != "$expected_kernel_commit" ]; then
   echo "Kernel commit mismatch: $actual_kernel_commit != $expected_kernel_commit" >&2
   exit 1
fi
if ! git -C "$kernel_source" merge-base --is-ancestor \
      "$BASE_KERNEL_COMMIT" "$actual_kernel_commit"; then
   echo "Kernel build does not contain required drm-agx3 base $BASE_KERNEL_COMMIT" >&2
   exit 1
fi
actual_driver_sha256=$(sha256sum "$driver_library" | awk '{print $1}')
if [ "$actual_driver_sha256" != "$expected_driver_sha256" ]; then
   echo "Driver digest mismatch: $actual_driver_sha256" >&2
   exit 1
fi
if ! grep -F "$driver_library" "$icd_file" >/dev/null; then
   echo "ICD does not name the verified driver library exactly" >&2
   exit 1
fi
actual_kernel_release=$(uname -r)
if [ "$actual_kernel_release" != "$expected_kernel_release" ]; then
   echo "Kernel release mismatch: $actual_kernel_release" >&2
   exit 1
fi

mkdir -p "$output_dir"
cache_dir=$(mktemp -d "$output_dir/cache.XXXXXX")
{
   echo "mesa_commit=$actual_mesa_commit"
   echo "mesa_base_commit=$BASE_MESA_COMMIT"
   echo "kernel_commit=$actual_kernel_commit"
   echo "kernel_base_commit=$BASE_KERNEL_COMMIT"
   echo "icd_file=$icd_file"
   echo "icd_sha256=$(sha256sum "$icd_file" | awk '{print $1}')"
   echo "driver_library=$driver_library"
   echo "driver_sha256=$actual_driver_sha256"
   echo "kernel_release=$actual_kernel_release"
   uname -a
} >"$output_dir/provenance.txt"
VK_DRIVER_FILES="$icd_file" vulkaninfo --summary \
   >"$output_dir/vulkan-summary.txt" 2>&1
if ! grep -F 'Apple A18 Pro' "$output_dir/vulkan-summary.txt" >/dev/null ||
   ! grep -Eiq 'Honeykrisp|MESA_HONEYKRISP' \
      "$output_dir/vulkan-summary.txt"; then
   echo "Honeykrisp G17P device missing from vulkaninfo" >&2
   exit 1
fi

active_pid=
cleanup()
{
   if [ -n "$active_pid" ] && kill -0 "$active_pid" 2>/dev/null; then
      kill -TERM "$active_pid" 2>/dev/null || true
      sleep 1
      kill -KILL "$active_pid" 2>/dev/null || true
   fi
}
trap cleanup EXIT INT TERM

image_diff_ppm()
{
   reference=$1
   actual=$2
   reference_size=$("$identify_program" -format '%wx%h' "$reference")
   actual_size=$("$identify_program" -format '%wx%h' "$actual")
   if [ "$reference_size" != "$actual_size" ]; then
      printf '%s\n' 1000000
      return
   fi
   pixels=$("$identify_program" -format '%[fx:w*h]' "$reference")
   set +e
   different=$("$compare_program" -metric AE -fuzz 2% \
      "$reference" "$actual" null: 2>&1)
   set -e
   different=$(printf '%s\n' "$different" | awk '{print $1}')
   awk -v different="$different" -v pixels="$pixels" \
      'BEGIN { printf "%u", (different * 1000000) / pixels }'
}

wait_for_image()
{
   reference=$1
   destination=$2
   deadline=$(( $(date +%s) + timeout_s ))
   while [ "$(date +%s)" -le "$deadline" ]; do
      if ! kill -0 "$active_pid" 2>/dev/null; then
         echo "SuperTuxKart exited before matching $reference" >&2
         return 1
      fi
      candidate="$destination.candidate.png"
      if "$capture_program" "$candidate" >/dev/null 2>&1; then
         ppm=$(image_diff_ppm "$reference" "$candidate")
         if [ "$ppm" -le "$visual_diff_ppm" ]; then
            mv "$candidate" "$destination"
            printf '%s\n' "$ppm"
            return 0
         fi
      fi
      sleep 0.25
   done
   echo "Timed out matching $reference" >&2
   return 1
}

summarize_submit_trace()
{
   awk '
      /G17P submit trace:/ {
         submissions++
         for (i = 1; i <= NF; ++i) {
            split($i, field, "=")
            if (field[1] == "vdm") vdm += field[2]
            if (field[1] == "cdm") cdm += field[2]
            if (field[1] == "draws") draws += field[2]
            if (field[1] == "dispatches") dispatches += field[2]
            if (field[1] == "ioctls") ioctls += field[2]
         }
      }
      END {
         printf "%u %u %u %u %u %u\n", submissions, vdm, cdm, draws,
                dispatches, ioctls
      }
   ' "$1"
}

run_one()
{
   label=$1
   max_menu_ms=$2
   shift 2
   run_dir="$output_dir/$label"
   config_dir=$(mktemp -d "$output_dir/config-$label.XXXXXX")
   mkdir -p "$run_dir"
   cp -a "$config_source/." "$config_dir/"
   journal_since=$(date --iso-8601=seconds)
   started_ns=$(date +%s%N)

   XDG_CONFIG_HOME="$config_dir" XDG_CACHE_HOME="$cache_dir" \
      VK_DRIVER_FILES="$icd_file" \
      HK_G17P_SUBMIT_TRACE=1 "$stk_bin" "$@" \
      >"$run_dir/stk.log" 2>&1 &
   active_pid=$!

   menu_ppm=$(wait_for_image "$menu_reference" "$run_dir/menu.png")
   menu_ns=$(date +%s%N)
   menu_ms=$(( (menu_ns - started_ns) / 1000000 ))
   if [ "$menu_ms" -gt "$max_menu_ms" ]; then
      echo "$label menu time ${menu_ms}ms exceeds ${max_menu_ms}ms" >&2
      return 1
   fi

   "$tutorial_hook" "$active_pid"
   tutorial_ppm=$(wait_for_image \
      "$tutorial_reference" "$run_dir/tutorial.png")

   kill -TERM "$active_pid" 2>/dev/null || true
   attempts=0
   while kill -0 "$active_pid" 2>/dev/null && [ "$attempts" -lt 20 ]; do
      sleep 0.25
      attempts=$((attempts + 1))
   done
   if kill -0 "$active_pid" 2>/dev/null; then
      kill -KILL "$active_pid" 2>/dev/null || true
   fi
   wait "$active_pid" 2>/dev/null || true
   active_pid=
   journalctl -k --since "$journal_since" >"$run_dir/kernel.log"

   if grep -Eiq 'asahi.*(fault|timed out|timeout|recovery|device lost|gmmu)|drm.*(gpu hang|gpu fault)' \
         "$run_dir/kernel.log"; then
      echo "$label recorded a kernel GPU fault or recovery" >&2
      return 1
   fi
   if grep -Eiq 'VK_ERROR_DEVICE_LOST|device[_ -]lost|DRM_IOCTL_ASAHI_SUBMIT failed' \
         "$run_dir/stk.log"; then
      echo "$label recorded Vulkan device loss" >&2
      return 1
   fi
   if ! grep -Eq 'OpenGL renderer: zink Vulkan .*MESA_HONEYKRISP' \
         "$run_dir/stk.log"; then
      echo "$label did not use Zink over Honeykrisp" >&2
      return 1
   fi

   set -- $(summarize_submit_trace "$run_dir/stk.log")
   submissions=$1; vdm=$2; cdm=$3; draws=$4; dispatches=$5; ioctls=$6
   if [ "$submissions" -eq 0 ] || [ "$vdm" -eq 0 ] || [ "$cdm" -eq 0 ]; then
      echo "$label is missing aggregate VDM/CDM trace coverage" >&2
      return 1
   fi
   if [ "$vdm" -gt "$max_vdm" ]; then
      echo "$label render stream count $vdm exceeds $max_vdm" >&2
      return 1
   fi
   if [ "$dispatches" -lt "$min_dispatches" ]; then
      echo "$label CDM launch count $dispatches is below $min_dispatches" >&2
      return 1
   fi

   printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$label" "$menu_ms" "$menu_ppm" "$tutorial_ppm" "$submissions" \
      "$vdm" "$draws" "$cdm" "$dispatches" "$ioctls" \
      >>"$output_dir/results.tsv"
}

printf 'run\tmenu_ms\tmenu_diff_ppm\ttutorial_diff_ppm\tsubmits\tvdm\tdraws\tcdm\tdispatches\tioctls\n' \
   >"$output_dir/results.tsv"
run_one cold "$max_cold_ms" "$@"
run_one warm "$max_warm_ms" "$@"

echo "G17P SuperTuxKart cold/warm acceptance PASS"
cat "$output_dir/results.tsv"

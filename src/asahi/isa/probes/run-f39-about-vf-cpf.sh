#!/bin/sh
set -eu

probe_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$probe_dir/../../../.." && pwd)
build_dir=$(CDPATH= cd -- "${1:-$repo_dir/_build-full}" && pwd)
vulkan_dir=$build_dir/src/asahi/vulkan
test_dir=$repo_dir/src/asahi/vulkan/tests
binary=$vulkan_dir/f39-about-vf-cpf
vertex_spv=$vulkan_dir/f39-about-vf-cpf.vert.spv
fragment_spv=$vulkan_dir/f39-about-vf-cpf.frag.spv
default_icd=$vulkan_dir/asahi_devenv_icd.aarch64.json
icd=${F39_VK_DRIVER_FILES:-${VK_DRIVER_FILES:-$default_icd}}
run_mode=${2:-}

timeout 30s cc -std=c11 -O2 -Wall -Wextra -Werror \
   -I"$build_dir/include" -I"$repo_dir/include" -I"$test_dir" \
   "$test_dir/g17p_vulkan_about_vf_cpf.c" -lvulkan -o "$binary"
timeout 30s glslangValidator -V \
   "$test_dir/g17p_vulkan_about_vf_cpf.vert" -o "$vertex_spv" >/dev/null
timeout 30s glslangValidator -V \
   "$test_dir/g17p_vulkan_about_vf_cpf.frag" -o "$fragment_spv" >/dev/null
timeout 30s spirv-val "$vertex_spv"
timeout 30s spirv-val "$fragment_spv"
timeout 10s "$binary" --check-spv "$vertex_spv" "$fragment_spv"

echo "f39 About VF_CPF: PASS — exact binary/SPIR-V bundle built in $vulkan_dir"
case "$run_mode" in
   "")
      echo "hardware execution skipped; pass --hardware-baseline or --hardware-flags1 on G17P"
      ;;
   --hardware-baseline|--hardware-flags1)
      if [ ! -r "$icd" ]; then
         echo "f39 About VF_CPF: ICD manifest is not readable: $icd" >&2
         exit 2
      fi
      trace_log=$(mktemp "${TMPDIR:-/tmp}/f39-about-vf-cpf.XXXXXX")
      trap 'rm -f "$trace_log"' EXIT HUP INT TERM
      if [ "$run_mode" = --hardware-flags1 ]; then
         flags1=1
         expected_tail=01000000
      else
         flags1=0
         expected_tail=00000000
      fi
      if ! env VK_DRIVER_FILES="$icd" HK_G17P_LAUNCH_TRACE=1 \
           HK_G17P_HAL200_ABOUT_FLAGS1="$flags1" \
           timeout 20s "$binary" >"$trace_log" 2>&1; then
         cat "$trace_log"
         echo "f39 About VF_CPF: hardware run failed or timed out" >&2
         exit 1
      fi
      cat "$trace_log"
      traced_fragment=$(grep 'G17P hal200-fragment-bind:' "$trace_log" | \
         tail -n 1 || true)
      case "$traced_fragment" in
      *"$expected_tail") ;;
      *)
         echo "f39 About VF_CPF: expected FragmentProgram tail $expected_tail was not traced" >&2
         exit 1
         ;;
      esac
      if [ "$run_mode" = --hardware-flags1 ] &&
         ! grep -q 'G17P HAL200 About flags=1 diagnostic active' "$trace_log"; then
         echo "f39 About VF_CPF: flags1 diagnostic did not activate" >&2
         exit 1
      fi
      echo "f39 About VF_CPF: $run_mode traced exact FragmentProgram tail $expected_tail"
      ;;
   *)
      echo "unknown option: $run_mode" >&2
      exit 2
      ;;
esac

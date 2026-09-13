#!/bin/sh
set -eu

# Usage: run-f48-hypr-mixed-draw.sh [GL_BUILD] [VK_BUILD]
#        [--software|--shim|--hardware]
#        [--tight|--api-barrier|--zink-sync]

test_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$test_dir/../../.." && pwd)
gl_build=$(CDPATH= cd -- "${1:-$repo_dir/_build-hypr-desktop}" && pwd)
vk_build=$(CDPATH= cd -- "${2:-$repo_dir/_build-full}" && pwd)
run_kind=${3:---shim}
sync_kind=${4:---tight}
binary=$gl_build/src/asahi/gl-tests/f48-hypr-mixed-draw
source=$test_dir/test-g17p-gl-hypr-mixed-draw.c
vertex=$test_dir/hypr-0.56.1-tex300.vert
fragment=$test_dir/hypr-0.56.1-border-rounded.frag
gallium=$gl_build/src/gallium/targets/dri/libgallium-26.1.4.so
egl_vendor=$gl_build/src/egl/libEGL_mesa.so.0.0.0
shim=$vk_build/src/asahi/drm-shim/libasahi_noop_drm_shim.so
icd=$vk_build/src/asahi/vulkan/asahi_devenv_icd.aarch64.json

case "$run_kind" in
   --software|--shim|--hardware) ;;
   *) echo "unknown run kind: $run_kind" >&2; exit 2 ;;
esac
case "$sync_kind" in
   --tight|--api-barrier|--zink-sync) ;;
   *) echo "unknown synchronization kind: $sync_kind" >&2; exit 2 ;;
esac
if [ "$run_kind" = --software ] && [ "$sync_kind" = --zink-sync ]; then
   echo "--zink-sync requires the Zink --shim or --hardware stack" >&2
   exit 2
fi

for file in "$source" "$vertex" "$fragment"; do
   test -r "$file" || { echo "missing F48 input: $file" >&2; exit 2; }
done

# Compile directly so this remains a standalone regression even when the
# selected Mesa build directory predates the test's optional Meson target.
mkdir -p "$(dirname -- "$binary")"
timeout -k 3s 60s cc -std=c11 -O2 -Wall -Wextra -Werror \
   $(pkg-config --cflags egl glesv2) "$source" -lEGL -lGLESv2 -o "$binary"

machine=$(file -b "$binary")
printf 'F48 binary: %s\n' "$machine"
printf '%s\n' "$machine" | grep -Eq 'ARM aarch64|ARM64' || {
   echo "F48 must be built as an AArch64 executable" >&2
   exit 2
}

program_mode=$sync_kind
if [ "$sync_kind" = --zink-sync ]; then
   program_mode=--tight
   export ZINK_DEBUG=sync
else
   unset ZINK_DEBUG
fi

if [ "$run_kind" = --software ]; then
   timeout -k 3s 30s env EGL_PLATFORM=surfaceless \
      LIBGL_ALWAYS_SOFTWARE=true \
      "$binary" "$vertex" "$fragment" "$program_mode"
   exit
fi

for file in "$gallium" "$egl_vendor" "$icd"; do
   test -r "$file" || { echo "missing F48 Zink input: $file" >&2; exit 2; }
done

runtime=$(mktemp -d)
cleanup()
{
   rm -f "$runtime/zink_dri.so" "$runtime/50_mesa.json" \
      "$runtime/f48-shim.log"
   rmdir "$runtime"
}
trap cleanup EXIT HUP INT TERM
ln -s "$gallium" "$runtime/zink_dri.so"
printf '{"file_format_version":"1.0.0","ICD":{"library_path":"%s"}}\n' \
   "$egl_vendor" >"$runtime/50_mesa.json"

export __EGL_VENDOR_LIBRARY_FILENAMES=$runtime/50_mesa.json
export LIBGL_DRIVERS_PATH=$runtime
export MESA_LOADER_DRIVER_OVERRIDE=zink
export MESA_SHADER_CACHE_DISABLE=true
export EGL_PLATFORM=surfaceless
export VK_DRIVER_FILES=$icd

if [ "$run_kind" = --hardware ]; then
   timeout -k 3s 30s "$binary" "$vertex" "$fragment" "$program_mode"
else
   test -r "$shim" || { echo "missing F48 no-op shim: $shim" >&2; exit 2; }
   export LD_PRELOAD=$shim
   export ASAHI_SHIM_CHIP=g17p
   export ASAHI_GL_NOOP_SHIM=1
   export ASAHI_SHIM_VALIDATE_G17P_STREAM=1
   shim_log=$runtime/f48-shim.log
   status=0
   timeout -k 3s 30s "$binary" "$vertex" "$fragment" "$program_mode" \
      >"$shim_log" 2>&1 || status=$?
   cat "$shim_log"
   if grep -Eiq 'invalid G17P command stream|VK_ERROR_DEVICE_LOST|device lost|GL error' \
        "$shim_log"; then
      echo "F48 no-op submission gate rejected driver/shim errors" >&2
      exit 1
   fi
   grep -F "F48 ${program_mode#--} no-op G17P submission validated" \
      "$shim_log" >/dev/null || {
      echo "F48 no-op submission gate did not see the success marker" >&2
      exit 1
   }
   exit "$status"
fi

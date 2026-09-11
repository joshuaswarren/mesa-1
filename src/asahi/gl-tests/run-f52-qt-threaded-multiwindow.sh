#!/bin/sh
set -eu

# Usage: run-f52-qt-threaded-multiwindow.sh [GL_BUILD] [VK_BUILD]
#        [--software|--shim|--hardware]
#        [--threaded-static|--threaded-growth]
#        [--overlapped|--serialized|--zink-sync]

test_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$test_dir/../../.." && pwd)
gl_build=$(CDPATH= cd -- "${1:-$repo_dir/_build-hypr-desktop}" && pwd)
vk_build=$(CDPATH= cd -- "${2:-$repo_dir/_build-full}" && pwd)
run_kind=${3:---shim}
workload=${4:---threaded-growth}
schedule=${5:---overlapped}
source=$test_dir/test-g17p-gl-qt-threaded-multiwindow.c
base_source=$test_dir/test-g17p-gl-qt-multiwindow-glyph-batch.c
native_vs=$test_dir/qt-6.11.2-textmask-compat.vert
native_fs=$test_dir/qt-6.11.2-textmask-alpha-compat.frag
sdf_vs=$test_dir/qt-6.11.2-distancefield-compat.vert
sdf_fs=$test_dir/qt-6.11.2-distancefield-alpha-compat.frag
binary=$gl_build/src/asahi/gl-tests/f52-qt-threaded-multiwindow
gallium=$gl_build/src/gallium/targets/dri/libgallium-26.1.4.so
egl_vendor=$gl_build/src/egl/libEGL_mesa.so.0.0.0
shim=$vk_build/src/asahi/drm-shim/libasahi_noop_drm_shim.so
icd=$vk_build/src/asahi/vulkan/asahi_devenv_icd.aarch64.json

case "$run_kind" in
   --software|--shim|--hardware) ;;
   *) echo "unknown run kind: $run_kind" >&2; exit 2 ;;
esac
case "$workload" in
   --threaded-static|--threaded-growth) ;;
   *) echo "unknown workload: $workload" >&2; exit 2 ;;
esac
case "$schedule" in
   --overlapped|--serialized|--zink-sync) ;;
   *) echo "unknown schedule: $schedule" >&2; exit 2 ;;
esac
if [ "$run_kind" = --software ] && [ "$schedule" = --zink-sync ]; then
   echo "--zink-sync requires the Zink --shim or --hardware stack" >&2
   exit 2
fi

for file in "$source" "$base_source" "$native_vs" "$native_fs" "$sdf_vs" "$sdf_fs"; do
   test -r "$file" || { echo "missing F52 input: $file" >&2; exit 2; }
done
mkdir -p "$(dirname -- "$binary")"
compile_flags=
compile_libs=
if [ "$run_kind" = --shim ]; then
   compile_flags=-DF52_SERIALIZE_NOOP_IOCTL
   compile_libs=-ldl
fi
timeout -k 3s 60s cc -std=c11 -O2 -Wall -Wextra -Werror \
   $compile_flags $(pkg-config --cflags egl gl) "$source" \
   -lEGL -lGL -lm $compile_libs -pthread -o "$binary"

machine=$(file -b "$binary")
printf 'F52 binary: %s\n' "$machine"
printf '%s\n' "$machine" | grep -Eq 'ARM aarch64|ARM64' || {
   echo "F52 must be built as an AArch64 executable" >&2
   exit 2
}

program_schedule=$schedule
if [ "$schedule" = --zink-sync ]; then
   program_schedule=--overlapped
   export ZINK_DEBUG=sync
else
   unset ZINK_DEBUG
fi
export MESA_GL_VERSION_OVERRIDE=2.1

if [ "$run_kind" = --software ]; then
   timeout -k 3s 30s env EGL_PLATFORM=surfaceless \
      LIBGL_ALWAYS_SOFTWARE=true "$binary" "$native_vs" "$native_fs" \
      "$sdf_vs" "$sdf_fs" "$workload" "$program_schedule"
   exit
fi

for file in "$gallium" "$egl_vendor" "$icd"; do
   test -r "$file" || { echo "missing F52 Zink input: $file" >&2; exit 2; }
done
runtime=$(mktemp -d)
cleanup()
{
   rm -f "$runtime/zink_dri.so" "$runtime/50_mesa.json" \
      "$runtime/f52-shim.log"
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
   timeout -k 3s 30s "$binary" "$native_vs" "$native_fs" \
      "$sdf_vs" "$sdf_fs" "$workload" "$program_schedule"
else
   test -r "$shim" || { echo "missing F52 no-op shim: $shim" >&2; exit 2; }
   export LD_PRELOAD=$shim
   export ASAHI_SHIM_CHIP=g17p
   export ASAHI_GL_NOOP_SHIM=1
   export ASAHI_SHIM_VALIDATE_G17P_STREAM=1
   shim_log=$runtime/f52-shim.log
   status=0
   timeout -k 3s 30s "$binary" "$native_vs" "$native_fs" \
      "$sdf_vs" "$sdf_fs" "$workload" "$program_schedule" \
      >"$shim_log" 2>&1 || status=$?
   cat "$shim_log"
   if grep -Eiq 'invalid G17P command stream|VK_ERROR_DEVICE_LOST|device lost|GL error|Segmentation fault|core dumped|Aborted|double free|corruption|realloc\(\)' \
        "$shim_log"; then
      echo "F52 no-op submission gate rejected driver/shim errors" >&2
      exit 1
   fi
   grep -F "F52 ${workload#--} ${program_schedule#--}:" "$shim_log" >/dev/null || {
      echo "F52 no-op submission gate did not see the success marker" >&2
      exit 1
   }
   grep -F 'G17P submission validated; pixels skipped' "$shim_log" >/dev/null || {
      echo "F52 no-op submission gate did not validate no-op completion" >&2
      exit 1
   }
   exit "$status"
fi

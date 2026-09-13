#!/bin/sh
set -eu

# Usage:
#   run-f46-hypr-shm-texture.sh [GL_BUILD] [VK_BUILD] \
#      [--software|--shim|--hardware] [--static|--updates|--resize]
#
# See README-f46-hypr-shm-texture.md for mode semantics and pass signatures.

test_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$test_dir/../../.." && pwd)
gl_build=$(CDPATH= cd -- "${1:-$repo_dir/_build-hypr-desktop}" && pwd)
vk_build=$(CDPATH= cd -- "${2:-$repo_dir/_build-full}" && pwd)
run_kind=${3:---shim}
mode=${4:---static}
output_dir=$gl_build/src/asahi/gl-tests
binary=$output_dir/f46-hypr-shm-texture
gallium=$gl_build/src/gallium/targets/dri/libgallium-26.1.4.so
egl_vendor=$gl_build/src/egl/libEGL_mesa.so.0.0.0
shim=$vk_build/src/asahi/drm-shim/libasahi_noop_drm_shim.so
icd=$vk_build/src/asahi/vulkan/asahi_devenv_icd.aarch64.json

case "$run_kind" in
   --shim|--hardware|--software) ;;
   *) echo "unknown run kind: $run_kind" >&2; exit 2 ;;
esac
case "$mode" in
   --static|--updates|--resize) ;;
   *) echo "unknown mode: $mode" >&2; exit 2 ;;
esac

for file in "$gallium" "$egl_vendor" "$icd" \
            "$test_dir/test-g17p-gl-hypr-shm-texture.c"; do
   test -r "$file" || { echo "missing f46 dependency: $file" >&2; exit 2; }
done

mkdir -p "$output_dir"
timeout -k 3s 60s cc -std=c11 -O2 -Wall -Wextra -Werror \
   $(pkg-config --cflags egl glesv2) \
   "$test_dir/test-g17p-gl-hypr-shm-texture.c" \
   -lEGL -lGLESv2 -o "$binary"

if [ "$run_kind" = --software ]; then
   timeout -k 3s 15s env EGL_PLATFORM=surfaceless \
      LIBGL_ALWAYS_SOFTWARE=true "$binary" "$mode"
   exit
fi

runtime=$(mktemp -d)
cleanup()
{
   rm -f "$runtime/zink_dri.so" "$runtime/50_mesa.json"
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
   timeout -k 3s 15s "$binary" "$mode"
else
   test -r "$shim" || { echo "missing f46 shim: $shim" >&2; exit 2; }
   export LD_PRELOAD=$shim
   export ASAHI_SHIM_CHIP=g17p
   export ASAHI_GL_NOOP_SHIM=1
   export ASAHI_SHIM_VALIDATE_G17P_STREAM=1
   timeout -k 3s 15s "$binary" "$mode"
fi

#!/bin/sh
set -eu

# Usage: run-f47-qt-scaled-glyph.sh [GL_BUILD] [VK_BUILD]
#        [--software|--shim|--hardware]

test_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$test_dir/../../.." && pwd)
gl_build=$(CDPATH= cd -- "${1:-$repo_dir/_build-hypr-desktop}" && pwd)
vk_build=$(CDPATH= cd -- "${2:-$repo_dir/_build-full}" && pwd)
run_kind=${3:---shim}
binary=$gl_build/src/asahi/gl-tests/f47-qt-scaled-glyph
gallium=$gl_build/src/gallium/targets/dri/libgallium-26.1.4.so
egl_vendor=$gl_build/src/egl/libEGL_mesa.so.0.0.0
shim=$vk_build/src/asahi/drm-shim/libasahi_noop_drm_shim.so
icd=$vk_build/src/asahi/vulkan/asahi_devenv_icd.aarch64.json

case "$run_kind" in
   --software|--shim|--hardware) ;;
   *) echo "unknown run kind: $run_kind" >&2; exit 2 ;;
esac

mkdir -p "$(dirname "$binary")"
timeout -k 3s 60s cc -std=c11 -O2 -Wall -Wextra -Werror \
   $(pkg-config --cflags egl glesv2) \
   "$test_dir/test-g17p-gl-qt-scaled-glyph.c" \
   -lEGL -lGLESv2 -lm -o "$binary"

if [ "$run_kind" = --software ]; then
   timeout -k 3s 20s env EGL_PLATFORM=surfaceless \
      LIBGL_ALWAYS_SOFTWARE=true "$binary"
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
   timeout -k 3s 20s "$binary"
else
   export LD_PRELOAD=$shim
   export ASAHI_SHIM_CHIP=g17p
   export ASAHI_GL_NOOP_SHIM=1
   export ASAHI_SHIM_VALIDATE_G17P_STREAM=1
   timeout -k 3s 20s "$binary"
fi

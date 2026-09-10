#!/bin/sh
# SPDX-License-Identifier: MIT
#
# Ask whether a wlr-screencopy SHM readback's (format, type) pair is accepted by
# this GL stack.  See the header of gl_shm_readback_format.c for why a rejected
# pair is silent: Hyprland's CGLFramebuffer::readPixels() never calls
# glGetError() and returns true regardless, so the client keeps a zero-filled
# buffer and a capture tool writes a fully transparent image.
#
# Usage:
#   run_gl_shm_readback_format.sh <mesa-build-dir> [--shim|--hardware|--software]
#
# --shim is meaningful even though the no-op driver executes nothing: the read
# format is validated in Mesa's state tracker, not on the GPU.
set -eu

test_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
build=$(CDPATH= cd -- "${1:?usage: $0 <mesa-build-dir> [--shim|--hardware|--software]}" && pwd)
kind=${2:---shim}

gallium=$build/src/gallium/targets/dri/libgallium-26.1.4.so
egl_vendor=$build/src/egl/libEGL_mesa.so.0.0.0
shim=$build/src/asahi/drm-shim/libasahi_noop_drm_shim.so
icd=$build/src/asahi/vulkan/asahi_devenv_icd.aarch64.json
bin=${TMPDIR:-/tmp}/gl_shm_readback_format

cc -std=c11 -O2 -Wall -o "$bin" "$test_dir/gl_shm_readback_format.c" \
   $(pkg-config --cflags egl glesv2) -lEGL -lGLESv2

if [ "$kind" = --software ]; then
   exec env EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=true "$bin"
fi

for file in "$gallium" "$egl_vendor" "$icd"; do
   test -r "$file" || { echo "missing dependency: $file" >&2; exit 2; }
done

runtime=$(mktemp -d)
cleanup() { rm -f "$runtime/zink_dri.so" "$runtime/50_mesa.json"; rmdir "$runtime"; }
trap cleanup EXIT HUP INT TERM
ln -s "$gallium" "$runtime/zink_dri.so"
printf '{"file_format_version":"1.0.0","ICD":{"library_path":"%s"}}\n' \
   "$egl_vendor" >"$runtime/50_mesa.json"

# Without this the loader can pick a system libgallium instead of this build.
export __EGL_VENDOR_LIBRARY_FILENAMES=$runtime/50_mesa.json
export LIBGL_DRIVERS_PATH=$runtime
export MESA_LOADER_DRIVER_OVERRIDE=zink
export MESA_SHADER_CACHE_DISABLE=true
export EGL_PLATFORM=surfaceless
export VK_DRIVER_FILES=$icd

if [ "$kind" = --hardware ]; then
   exec "$bin"
fi

test -r "$shim" || { echo "missing shim: $shim" >&2; exit 2; }
export LD_PRELOAD=$shim
export ASAHI_SHIM_CHIP=g17p
export ASAHI_GL_NOOP_SHIM=1
export ASAHI_SHIM_VALIDATE_G17P_STREAM=1
exec "$bin"

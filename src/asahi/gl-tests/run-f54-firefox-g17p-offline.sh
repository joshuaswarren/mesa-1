#!/bin/sh
set -eu

# Usage: run-f54-firefox-g17p-offline.sh [GL_BUILD] [VK_BUILD]
#                                  [--software|--shim|--compiler]

test_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$test_dir/../../.." && pwd)
gl_build=$(CDPATH= cd -- "${1:-$repo_dir/_build-hypr-desktop}" && pwd)
vk_build=$(CDPATH= cd -- "${2:-$repo_dir/_build-full}" && pwd)
run_kind=${3:---shim}
source=$test_dir/test-g17p-gl-firefox-webgl.c
binary=$gl_build/src/asahi/gl-tests/f54-firefox-webgl
gallium=$gl_build/src/gallium/targets/dri/libgallium-26.1.4.so
egl_vendor=$gl_build/src/egl/libEGL_mesa.so.0.0.0
shim=$vk_build/src/asahi/drm-shim/libasahi_noop_drm_shim.so
icd=$vk_build/src/asahi/vulkan/asahi_devenv_icd.aarch64.json

case "$run_kind" in
   --software|--shim|--compiler) ;;
   *) echo "unknown F54 run kind: $run_kind" >&2; exit 2 ;;
esac

test -r "$source" || { echo "missing F54 source: $source" >&2; exit 2; }

if [ "$run_kind" = --compiler ]; then
   timeout -k 3s 90s meson compile -C "$vk_build" \
      agx3_graphics_compile_test agx3_pack_test agx3_alu_core_test \
      agx3_iadd_core_test agx3_graphics_program_test
   compiler_dir=$vk_build/src/asahi/compiler
   timeout -k 3s 30s "$compiler_dir/agx3_iadd_core_test"
   timeout -k 3s 30s "$compiler_dir/agx3_alu_core_test"
   timeout -k 3s 30s "$compiler_dir/agx3_graphics_program_test"
   timeout -k 3s 45s "$compiler_dir/agx3_graphics_compile_test"
   timeout -k 3s 45s "$compiler_dir/agx3_pack_test" \
      '--gtest_filter=*Firefox*:*VertexIndexedTexture*'
   echo "F54 PASS: focused Firefox/G17P compiler gates"
   exit
fi

mkdir -p "$(dirname -- "$binary")"
timeout -k 3s 60s cc -std=c11 -O2 -Wall -Wextra -Werror \
   $(pkg-config --cflags egl glesv2) "$source" -lEGL -lGLESv2 -o "$binary"

machine=$(file -b "$binary")
printf 'F54 binary: %s\n' "$machine"
printf '%s\n' "$machine" | grep -Eq 'ARM aarch64|ARM64' || {
   echo "F54 must be built as an AArch64 executable" >&2
   exit 2
}

if [ "$run_kind" = --software ]; then
   timeout -k 3s 30s env EGL_PLATFORM=surfaceless \
      LIBGL_ALWAYS_SOFTWARE=true "$binary"
   exit
fi

for file in "$gallium" "$egl_vendor" "$shim" "$icd"; do
   test -r "$file" || { echo "missing F54 no-op input: $file" >&2; exit 2; }
done

runtime=$(mktemp -d)
cleanup()
{
   rm -f "$runtime/zink_dri.so" "$runtime/50_mesa.json" \
      "$runtime/f54-shim.log"
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
export LD_PRELOAD=$shim
export ASAHI_SHIM_CHIP=g17p
export ASAHI_GL_NOOP_SHIM=1
export ASAHI_SHIM_VALIDATE_G17P_STREAM=1

log=$runtime/f54-shim.log
status=0
timeout -k 3s 30s "$binary" >"$log" 2>&1 || status=$?
cat "$log"
if [ "$status" -ne 0 ]; then
   echo "F54 no-op executable failed with status $status" >&2
   exit "$status"
fi
if grep -Eiq 'invalid G17P command stream|VK_ERROR_DEVICE_LOST|device lost|GL error|Segmentation fault|core dumped|Aborted|double free|corruption|cannot encode|no grounded AGX3 encoding|pipeline creation failed' "$log"; then
   echo "F54 no-op submission gate rejected driver/shim errors" >&2
   exit 1
fi
grep -F 'F54 PASS: G17P Firefox WebGL submission validated; pixels skipped' \
   "$log" >/dev/null || {
   echo "F54 no-op submission gate did not see its success marker" >&2
   exit 1
}
indexed=$(grep -c 'G17P VDM validated: 1 draws, 1 indexed' "$log" || true)
if [ "$indexed" -ne 13 ]; then
   echo "F54 expected exactly 13 indexed VDM validations, saw $indexed" >&2
   exit 1
fi
printf 'F54 no-op gate: exactly %s indexed submissions validated\n' "$indexed"

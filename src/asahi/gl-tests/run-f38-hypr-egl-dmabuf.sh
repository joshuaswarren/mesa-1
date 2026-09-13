#!/bin/sh
set -eu

test_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$test_dir/../../.." && pwd)
gl_build=$(CDPATH= cd -- "${1:-$repo_dir/_build-hypr-desktop}" && pwd)
vk_build=$(CDPATH= cd -- "${2:-$repo_dir/_build-full}" && pwd)
run_hardware=${3:-}
output_dir=$gl_build/src/asahi/gl-tests
binary=$output_dir/f38-hypr-egl-dmabuf
gallium=$gl_build/src/gallium/targets/dri/libgallium-26.1.4.so
native_gallium=$repo_dir/_build-gallium/src/gallium/targets/dri/libgallium-26.1.4.so
egl_vendor=$gl_build/src/egl/libEGL_mesa.so.0.0.0
gbm_backend=$gl_build/src/gbm/backends/dri
shim=$vk_build/src/asahi/drm-shim/libasahi_noop_drm_shim.so
icd=$vk_build/src/asahi/vulkan/asahi_devenv_icd.aarch64.json

for file in "$gallium" "$native_gallium" "$egl_vendor" "$icd" \
            "$test_dir/test-g17p-egl-dmabuf-hypr.c"; do
   test -r "$file" || { echo "missing f38 dependency: $file" >&2; exit 2; }
done
mkdir -p "$output_dir"
cc -std=c11 -O2 -Wall -Wextra -Werror -D_GNU_SOURCE \
   $(pkg-config --cflags gbm libdrm egl glesv2) \
   -I"$repo_dir/include" -I"$gl_build/include" \
   "$test_dir/test-g17p-egl-dmabuf-hypr.c" \
   -L"$gl_build/src/gbm" -Wl,-rpath,"$gl_build/src/gbm" \
   -lgbm -lEGL -lGLESv2 -ldrm -lvulkan -o "$binary"

runtime=$(mktemp -d)
cleanup()
{
   rm -f "$runtime/zink_dri.so" "$runtime/asahi_dri.so" \
      "$runtime/50_mesa.json"
   rmdir "$runtime"
}
trap cleanup EXIT HUP INT TERM
ln -s "$gallium" "$runtime/zink_dri.so"
ln -s "$native_gallium" "$runtime/asahi_dri.so"
printf '{"file_format_version":"1.0.0","ICD":{"library_path":"%s"}}\n' \
   "$egl_vendor" >"$runtime/50_mesa.json"

export __EGL_VENDOR_LIBRARY_FILENAMES=$runtime/50_mesa.json
export LIBGL_DRIVERS_PATH=$runtime
export GBM_BACKENDS_PATH=$gbm_backend
export MESA_LOADER_DRIVER_OVERRIDE=zink
export EGL_PLATFORM=surfaceless
export VK_DRIVER_FILES=$icd

if [ "$run_hardware" = --hardware ]; then
   "$binary"
elif [ -n "$run_hardware" ]; then
   echo "unknown option: $run_hardware (expected --hardware)" >&2
   exit 2
else
   test -r "$shim" || { echo "missing f38 shim: $shim" >&2; exit 2; }
   export LD_PRELOAD=$shim
   export ASAHI_SHIM_CHIP=g17p
   export ASAHI_GL_NOOP_SHIM=1
   export ASAHI_SHIM_VALIDATE_G17P_STREAM=1
   "$binary"
fi

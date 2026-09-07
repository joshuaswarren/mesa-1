#!/bin/sh
# SPDX-License-Identifier: MIT
#
# Build and run the GL fill/readback oracle against an installed GL stack, over
# a size sweep. The sweep matters: the G17P failure that motivated this test is
# size-dependent (a 16x16 RGBA8 image->buffer roundtrip is qualified exact in
# VULKAN_FEATURES.md row X05b, while 512x512 through GL reads back as zeros), so
# a single size cannot tell a tiling/addressing defect from a transfer-size one.
#
# Usage:
#   run_gl_fbo_readback.sh <mesa-install-lib-dir> [sizes...]
#
# The software control, which must pass on any correct driver:
#   NEO_SURFACELESS=1 LIBGL_ALWAYS_SOFTWARE=1 run_gl_fbo_readback.sh <dir>
set -eu

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
libdir=${1:?usage: $0 <mesa-install-lib-dir> [WxH ...]}
shift || true

if [ "$#" -eq 0 ]; then
   set -- 16x16 64x64 512x512 1024x1024 2408x1506
fi

bin=${TMPDIR:-/tmp}/gl_fbo_readback
cc -O2 -g -Wall -o "$bin" "$script_dir/gl_fbo_readback.c" \
   -lEGL -lGLESv2 -L"$libdir" -lgbm -lm

export LD_LIBRARY_PATH="$libdir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
: "${GBM_BACKENDS_PATH:=$libdir/gbm}"
export GBM_BACKENDS_PATH

rc=0
for size in "$@"; do
   w=${size%x*}
   h=${size#*x}
   echo "=== ${w}x${h}"
   "$bin" "$w" "$h" || rc=1
done

exit "$rc"

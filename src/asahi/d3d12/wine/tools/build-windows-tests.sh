#!/bin/bash
# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT
#
# Cross-build the frozen d3d12agx suites as x86-64 Windows executables (what a
# game is under Proton), plus the HelloTriangle sample, with llvm-mingw.
#
# Usage: build-windows-tests.sh <mesa-source> <out-dir> [dx-samples-src]
set -euo pipefail
src=$(cd "$1" && pwd); mkdir -p "$2"; out=$(cd "$2" && pwd); samples=${3:-}
toolchain=${LLVM_MINGW_ROOT:-/opt/aurora-llvm-mingw-20260922/llvm-mingw-20260922-ucrt-ubuntu-22.04-aarch64}
cc=$toolchain/bin/x86_64-w64-mingw32-clang
cxx=$toolchain/bin/x86_64-w64-mingw32-clang++
tests=$src/src/asahi/d3d12/tests
dxh=$src/subprojects/DirectX-Headers-1.0/include
[[ -d $dxh ]] || dxh=$(ls -d "$src"/subprojects/DirectX-Headers*/include | head -1)
mkdir -p "$out/suites" "$out/samples"

# -static: llvm-mingw otherwise links libc++/libunwind as DLLs that no Wine
# prefix has, and the exe dies at import time (STATUS_DLL_NOT_FOUND).
cflags=(-O2 -std=c11 -D_WIN32_WINNT=0x0A00 -DINITGUID -Wno-ignored-attributes -static
        -I"$dxh" -I"$src/src/asahi/d3d12" -I"$src/src/asahi/d3d12/tests")
for name in g1_smoke g2_smoke g3_smoke g4_smoke g4_triangle g5_depth \
            g5_descriptors g5_atomics g5_control_flow g5_wave g5_texture \
            g6_geometry g7_tessellation g8_mesh g9_amplification \
            g10_vertex_buffers g11_indexed_instanced; do
   "$cc" "${cflags[@]}" -o "$out/suites/d3d12agx_$name.exe" "$tests/$name.c" -ld3d12 -ldxguid -lole32
   echo "built $name.exe"
done
cp -r "$tests/shaders" "$out/suites/"
cp "$tests/run-suites.sh" "$out/suites/"

if [[ -n $samples ]]; then
   hw=$samples/Samples/Desktop/D3D12HelloWorld/src
   wrap=$src/src/asahi/d3d12/wine/samples
   # mingw's own d3d12/dxgi headers: with WIDL_EXPLICIT_AGGREGATE_RETURNS the
   # C++ wrappers use the explicit out-pointer ABI for aggregate returns, which
   # is what d3d12agx (and vkd3d-proton) implement.
   for name in HelloTriangle HelloTexture HelloBundles HelloConstBuffers; do
      cls=D3D12$name
      # Every HelloWorld sample in this DirectX-Graphics-Samples revision
      # reads shaders_VSMain.cso / shaders_PSMain.cso that Visual Studio
      # compiles at build time; the harness produces them at start-up.
      extra=-DSAMPLE_PRECOMPILED
      mkdir -p "$out/samples/$cls"
      "$cxx" -O2 -std=c++17 -static -municode -D_WIN32_WINNT=0x0A00 -DUNICODE -D_UNICODE -DNOMINMAX \
         -DWIDL_EXPLICIT_AGGREGATE_RETURNS "-D_uuidof=__uuidof" \
         -DSAMPLE_CLASS=$cls "-DSAMPLE_HEADER=\"$cls.h\"" "${extra[@]}" \
         -Wno-ignored-attributes -Wno-nonportable-include-path -Wno-address-of-temporary \
         -I"$wrap" -I"$hw/$name" \
         -o "$out/samples/$cls/$cls.exe" \
         "$wrap/aurora_sample_main.cpp" "$hw/$name/$cls.cpp" "$hw/$name/DXSample.cpp" "$hw/$name/Win32Application.cpp" \
         -ld3d12 -ldxgi -ld3dcompiler -ldxguid -lole32 -luser32 -lgdi32
      # each sample reads its own shaders.hlsl next to the exe (d3dcompiler_47)
      cp "$hw/$name/shaders.hlsl" "$out/samples/$cls/"
      echo "built $cls.exe"
   done
fi
(cd "$out" && find . -type f -name '*.exe' -exec sha256sum {} + > ARTIFACTS.sha256)

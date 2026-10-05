#!/bin/bash
# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT
#
# Cross-build the frozen d3d12agx suites as x86-64 Windows executables (what a
# game is under Proton), plus the HelloTriangle sample, with llvm-mingw.
#
# Usage: build-windows-tests.sh <mesa-source> <out-dir> [dx-samples-src]
set -euo pipefail
src=$(realpath "$1"); out=$(realpath -m "$2"); samples=${3:-}
toolchain=${LLVM_MINGW_ROOT:-/opt/aurora-llvm-mingw-20260922/llvm-mingw-20260922-ucrt-ubuntu-22.04-aarch64}
cc=$toolchain/bin/x86_64-w64-mingw32-clang
cxx=$toolchain/bin/x86_64-w64-mingw32-clang++
tests=$src/src/asahi/d3d12/tests
dxh=$src/subprojects/DirectX-Headers-1.0/include
[[ -d $dxh ]] || dxh=$(ls -d "$src"/subprojects/DirectX-Headers*/include | head -1)
mkdir -p "$out/suites" "$out/samples"

cflags=(-O2 -std=c11 -D_WIN32_WINNT=0x0A00 -DINITGUID -Wno-ignored-attributes
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
   ht=$samples/Samples/Desktop/D3D12HelloWorld/src/HelloTriangle
   wrap=$src/src/asahi/d3d12/wine/samples
   "$cxx" -O2 -std=c++17 -municode -D_WIN32_WINNT=0x0A00 -DUNICODE -D_UNICODE -DNOMINMAX \
      -Wno-ignored-attributes -Wno-nonportable-include-path \
      -I"$dxh" -I"$dxh/directx" -I"$ht" -I"$wrap" \
      -o "$out/samples/D3D12HelloTriangle.exe" \
      "$wrap/aurora_sample_main.cpp" "$ht/D3D12HelloTriangle.cpp" "$ht/DXSample.cpp" "$ht/Win32Application.cpp" \
      -ld3d12 -ldxgi -ld3dcompiler -ldxguid -lole32 -luser32 -lgdi32
   cp "$ht/shaders.hlsl" "$out/samples/"
   echo "built D3D12HelloTriangle.exe"
fi
(cd "$out" && find . -type f -name '*.exe' -exec sha256sum {} + > ARTIFACTS.sha256)

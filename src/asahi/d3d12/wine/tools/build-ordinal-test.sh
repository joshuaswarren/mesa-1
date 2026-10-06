#!/bin/bash
# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT
# Usage: build-ordinal-test.sh <mesa-source> <out-dir> (Linux build host)
set -euo pipefail
src=$(realpath "$1")
mkdir -p "$2"
out=$(realpath "$2")
tc=${LLVM_MINGW_ROOT:?set LLVM_MINGW_ROOT to llvm-mingw}
tests=$src/src/asahi/d3d12/wine/tests
dxh=$src/subprojects/DirectX-Headers-1.0/include
"$tc/bin/llvm-dlltool" -m i386:x86-64 -d "$tests/d3d12-ordinal-import.def" \
   -l "$out/libd3d12-ordinal.a"
"$tc/bin/x86_64-w64-mingw32-clang" -O2 -std=c11 -static -DINITGUID \
   -D_WIN32_WINNT=0x0A00 -I"$dxh" "$tests/ordinal-agility.c" \
   "$out/libd3d12-ordinal.a" -o "$out/ordinal-agility.exe"
"$tc/bin/llvm-readobj" --coff-imports --coff-exports "$out/ordinal-agility.exe" \
   > "$out/ordinal-agility.pe.txt"
cat "$out/ordinal-agility.pe.txt"
python3 - "$out/ordinal-agility.pe.txt" <<'PY'
import re
import sys
dump = open(sys.argv[1]).read()
block = next(b for b in re.findall(r'Import \{(.*?)\}', dump, re.S)
             if re.search(r'Name: d3d12\.dll\s', b))
symbols = re.findall(r'Symbol: (.*?)\n', block)
expected = [' (101)', ' (102)', 'D3D12GetInterface (0)',
            'D3D12SerializeRootSignature (0)',
            'D3D12SerializeVersionedRootSignature (0)']
if symbols != expected:
    sys.exit('FAIL: the probe must import 101/102 by ordinal and serializers by name')
print('PASS: x64 ordinal imports 101/102 and named Agility/serializer imports')
PY
mkdir -p "$out/D3D12/x64"
"$tc/bin/x86_64-w64-mingw32-clang" -shared -O2 -static \
   "$tests/agility-core-trap.c" -o "$out/D3D12/x64/D3D12Core.dll"

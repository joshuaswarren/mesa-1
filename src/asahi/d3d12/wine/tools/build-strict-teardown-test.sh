#!/bin/bash
# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT
# Usage: build-strict-teardown-test.sh <mesa-source> <out-dir> (Linux host)
set -euo pipefail
src=$(realpath "$1")
mkdir -p "$2"
out=$(realpath "$2")
tc=${LLVM_MINGW_ROOT:?set LLVM_MINGW_ROOT to llvm-mingw}
tests=$src/src/asahi/d3d12/wine/tests
"$tc/bin/x86_64-w64-mingw32-clang" -O2 -std=c11 -static -DINITGUID \
   -D_WIN32_WINNT=0x0A00 -I"$src/subprojects/DirectX-Headers-1.0/include" \
   "$tests/strict-teardown.c" -ld3d12 \
   -o "$out/strict-teardown.exe"
"$tc/bin/llvm-readobj" --coff-imports "$out/strict-teardown.exe" \
   > "$out/strict-teardown.pe.txt"
# A handler installed in a private CRT would not catch the driver's errors.
/usr/bin/grep -q '_set_invalid_parameter_handler' "$out/strict-teardown.pe.txt"
/usr/bin/grep -q 'api-ms-win-crt-runtime' "$out/strict-teardown.pe.txt"
cat "$out/strict-teardown.pe.txt"

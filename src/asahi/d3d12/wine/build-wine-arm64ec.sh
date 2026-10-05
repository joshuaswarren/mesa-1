#!/bin/bash
# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT
#
# Reproducible build of the Wine/Proton d3d12agx pair on an aarch64 Linux box
# (the Neo) with llvm-mingw:
#   1. native Mesa host tools (mesa_clc, vtn_bindgen2, asahi_clc) from this tree
#   2. ARM64EC PE d3d12.dll (meson cross build, -Dasahi-d3d12-platform=wine)
#   3. aarch64 ELF unixlib d3d12.so
#   4. Wine builtin marking + SHA256 manifest
#
# Usage: build-wine-arm64ec.sh <mesa-source> <out-dir> [jobs]
# Env:   LLVM_MINGW_ROOT (default /opt/aurora-llvm-mingw-20260922/llvm-mingw-20260922-ucrt-ubuntu-22.04-aarch64)
#        NATIVE_BUILD (reuse an existing native build dir with the host tools)
set -euo pipefail

src=$(realpath "$1")
out=$(realpath -m "$2")
jobs=${3:-2}
toolchain=${LLVM_MINGW_ROOT:-/opt/aurora-llvm-mingw-20260922/llvm-mingw-20260922-ucrt-ubuntu-22.04-aarch64}
native=${NATIVE_BUILD:-$out/native-tools}
cross=$out/cross-arm64ec
wine_dir=$src/src/asahi/d3d12/wine

for tool in "$toolchain/bin/arm64ec-w64-mingw32-clang" "$toolchain/bin/llvm-readobj" meson ninja cc python3; do
   if [[ $tool == */* ]]; then [[ -x $tool ]] || { echo "missing $tool" >&2; exit 2; }
   else command -v "$tool" >/dev/null || { echo "missing $tool" >&2; exit 2; }; fi
done
mkdir -p "$out"
exec > >(tee -a "$out/BUILD.log") 2>&1
echo "== $(date -u +%FT%TZ) source $(git -C "$src" rev-parse HEAD 2>/dev/null || cat "$src/SOURCE-COMMIT")"

# 1. native host tools ------------------------------------------------------
if [[ ! -x $native/src/compiler/clc/mesa_clc || ! -x $native/src/asahi/clc/asahi_clc ]]; then
   echo "== native host tools -> $native"
   meson setup "$native" "$src" \
      -Dplatforms= -Dvulkan-drivers= -Dgallium-drivers= -Dtools=asahi \
      -Dllvm=enabled -Dbuild-tests=false -Dvideo-codecs= -Dglx=disabled \
      -Degl=disabled -Dgbm=disabled -Dshared-glapi=disabled \
      -Dmesa-clc=enabled -Dprecomp-compiler=enabled -Dbuildtype=release \
      ${NATIVE_EXTRA_ARGS:-}
   ninja -C "$native" -j"$jobs" src/compiler/clc/mesa_clc src/compiler/spirv/vtn_bindgen2 src/asahi/clc/asahi_clc
fi
export PATH=$native/src/compiler/clc:$native/src/compiler/spirv:$native/src/asahi/clc:$PATH

# 2. ARM64EC PE -------------------------------------------------------------
cross_file=$out/windows-arm64ec-cross.ini
sed -e "s#@TOOLCHAIN@#$toolchain#g" -e "s#@WINE_INCLUDE@#$wine_dir/include#g" \
   "$wine_dir/tools/windows-arm64ec-cross.ini.in" > "$cross_file"
echo "== cross build -> $cross"
cross_args=(
   --cross-file "$cross_file"
   -Dbuildtype=release -Db_ndebug=true -Dwarning_level=1 -Dwerror=false
   -Dplatforms=windows -Dvulkan-drivers= -Dgallium-drivers=
   -Dasahi-d3d12=true -Dasahi-d3d12-platform=wine
   -Dtools= -Dllvm=disabled -Dglx=disabled -Degl=disabled -Dgbm=disabled
   -Dshared-glapi=disabled -Dbuild-tests=false -Dvideo-codecs=
   -Dmesa-clc=system -Dprecomp-compiler=system -Dmin-windows-version=10
   -Dzlib=disabled -Dshader-cache=disabled -Dvalgrind=disabled
   ${CROSS_EXTRA_ARGS:-}
)
if [[ -f $cross/build.ninja ]]; then
   meson setup --reconfigure "$cross" "$src" "${cross_args[@]}"
else
   rm -rf "$cross"
   meson setup "$cross" "$src" "${cross_args[@]}"
fi
ninja -C "$cross" -j"$jobs" src/asahi/d3d12/d3d12.dll

# 3. unixlib ----------------------------------------------------------------
echo "== unixlib"
cc -O2 -fPIC -shared -fvisibility=hidden -Wall -Wextra \
   -o "$out/d3d12.so" "$wine_dir/unix/d12_wine_unix.c"

# 4. package ----------------------------------------------------------------
cp "$cross/src/asahi/d3d12/d3d12.dll" "$out/d3d12.dll"
python3 "$wine_dir/tools/mark-wine-builtin.py" "$out/d3d12.dll"
"$toolchain/bin/llvm-readobj" --file-headers --coff-load-config "$out/d3d12.dll" \
   | grep -E 'Format|Machine|CHPEMetadataPointer' > "$out/d3d12.dll.readobj.txt"
cat "$out/d3d12.dll.readobj.txt"
# llvm-readobj reports ARM64EC for an AMD64-machine image with CHPE metadata;
# mark-wine-builtin.py verified the raw header field is 0x8664 (what
# wineserver requires).
grep -q 'IMAGE_FILE_MACHINE_ARM64EC\|IMAGE_FILE_MACHINE_AMD64' "$out/d3d12.dll.readobj.txt"
grep -q 'CHPEMetadataPointer: 0x[1-9A-F]' "$out/d3d12.dll.readobj.txt" || { echo "not an ARM64EC image" >&2; exit 1; }
git -C "$src" rev-parse HEAD > "$out/SOURCE-COMMIT" 2>/dev/null || cp "$src/SOURCE-COMMIT" "$out/SOURCE-COMMIT"
(cd "$out" && sha256sum d3d12.dll d3d12.so > ARTIFACTS.sha256 && cat ARTIFACTS.sha256)
echo "== done $(date -u +%FT%TZ)"

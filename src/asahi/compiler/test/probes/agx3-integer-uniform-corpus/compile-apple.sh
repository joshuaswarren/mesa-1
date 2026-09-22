#!/bin/sh
# SPDX-License-Identifier: MIT
set -eu
src=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
out=${1:?usage: compile-apple.sh OUTPUT_DIRECTORY}
mkdir -p "$out"
out=$(CDPATH= cd -- "$out" && pwd)
xcrun metal --version > "$out/toolchain.txt"
xcrun metal -c -fno-fast-math "$src/kernels.metal" -o "$out/kernels.air"
xcrun metallib "$out/kernels.air" -o "$out/kernels.metallib"
for arch in applegpu_g17p applegpu_g16g; do
    mkdir -p "$out/$arch"
    while IFS= read -r name; do
        [ -n "$name" ] || continue
        base="$out/$arch/$name"
        printf '{"pipelines":{"compute_pipelines":[{"compute_function":"%s"}]}}\n' "$name" > "$base.mtlp-json"
        xcrun metal-nt -arch "$arch" -platform_version macos 27.0 27.0 \
            -N "$base.mtlp-json" "$out/kernels.metallib" -o "$base.bin"
        shasum -a 256 "$base.bin"
    done < "$src/cases.txt"
done

#!/usr/bin/env bash
set -e
SRC_DIR=${1:-.}; BUILD_DIR=${2:-$SRC_DIR/build-trig}; cd "$SRC_DIR"
cd "$BUILD_DIR"
ninja -t commands | grep -m1 "nir.c.o" | sed 's/.*cc //;s/ [^ ]*\.c\.o.*//;s/-mtls-dialect=[^ ]*//' | tr ' ' '\n' | grep -E "^-I|^-D|^-std|^-pthread|^-fvisibility" | grep -v "^-MF$|^-MQ$" | tr '\n' ' ' > /tmp/flags.raw
cd ..
{
  printf 'cc /tmp/spvstub.c trig_dump.c -o trig_dump '
  printf '%s ' -Ibuild-trig/src -Ibuild-trig/src/asahi/compiler -Ibuild-trig/src/compiler
  find build-trig/src -type d -printf "-I%p "
  printf '%s ' -Iinclude -Isrc -Isrc/asahi -Isrc/asahi/compiler -Isrc/asahi/isa -Isrc/compiler/nir -Isrc/compiler/spirv -Isrc/compiler -Isrc/util -Isrc/gallium/include
  printf '%s ' build-trig/src/compiler/spirv/libvtn.a build-trig/src/asahi/isa/libagx2_disasm.a build-trig/src/asahi/compiler/libasahi_compiler.a build-trig/src/compiler/nir/libnir.a build-trig/src/util/libmesa_util.a build-trig/src/c11/impl/libmesa_util_c11.a build-trig/src/compiler/libcompiler.a -lm -lpthread build-trig/src/util/blake3/libblake3.a
  cat /tmp/flags.raw
} > /tmp/run.sh
echo >> /tmp/run.sh
bash /tmp/run.sh 2>&1 | grep -E "error|Error" | head -5
ls -la trig_dump

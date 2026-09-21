#!/usr/bin/env bash
# Copyright 2026 Joshua Warren
# SPDX-License-Identifier: MIT
# 06 - trig lowering invariance test
#
# Compiles 06-trig-invariance.comp through the same NIR pipeline
# the Honeykrisp driver uses (offline via ../../trig_dump.c) and
# asserts that the AGX disassembly matches
# 06-trig-invariance.isa.golden. The golden must be captured on
# the reference hardware-under-test before this test has any
# invariant meaning.
#
# Maintained by the agent/qual-pd-submit-overhead lane as the tested
# root-cause lever for Mesa honeykrisp trig lowerings.
set -euo pipefail
HERE=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)
TRIG_DUMP=${TRIG_DUMP:-$HERE/trig_dump}
GLSLANG=${GLSLANG:-glslangValidator}
SPV=$HERE/06-trig-invariance.spv
DUMP_DIR=$HERE/06-trig-invariance.dump
GOLDEN=$HERE/06-trig-invariance.isa.golden

[ -x "$TRIG_DUMP" ] || { echo "error: trig_dump not built at $TRIG_DUMP" >&2; exit 2; }
"$GLSLANG" -V "$HERE/06-trig-invariance.comp" -o "$SPV"
mkdir -p "$DUMP_DIR"
TRIG_DUMP_DIR="$DUMP_DIR" "$TRIG_DUMP" "$SPV" > "$DUMP_DIR/stdout.txt" 2>&1

# First run? Capture golden. If already exists, diff.
if [ ! -f "$GOLDEN" ]; then
  echo "warning: golden missing; first run captures it"
  cp "$DUMP_DIR/isa.txt" "$GOLDEN"
  exit 0
fi

if ! diff -u "$GOLDEN" "$DUMP_DIR/isa.txt" > "$DUMP_DIR/diff.txt"; then
  echo "TRIG INVARIANCE FAILED"
  cat "$DUMP_DIR/diff.txt"
  exit 1
fi
echo "TRIG INVARIANCE PASS"
exit 0

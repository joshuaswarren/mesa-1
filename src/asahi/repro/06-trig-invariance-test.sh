#!/usr/bin/env bash
# Copyright 2026 Joshua Warren
# SPDX-License-Identifier: MIT
# 06 - trig lowering invariance test (strict)
#
# Modes:
#   --capture   Explicitly write the golden file from the current isa.txt
#              output. Only valid on a known-good reference hardware-
#              under-test. The default mode (no flag) DIFFS against the
#              golden and EXITS 1 on any mismatch; it EXITS 2 if the
#              golden file is missing (NOT a pass-on-missing gate).
#
# Captures the AGX disassembly via ../../trig_dump.c (commit d1fed280ec2
# on hk/trig-invariance) for the 06-trig-invariance.comp fixture.
#
# Acceptance gate is FAIL-on-mismatch, NOT pass-on-missing. To prove
# the test catches regressions, see 06-trig-invariance-perturb.sh which
# applies a one-line perturbation to a local copy of the relevant Mesa
# file and re-runs the test to verify it fails.

set -euo pipefail
HERE=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)
TRIG_DUMP=${TRIG_DUMP:-$HERE/trig_dump}
GLSLANG=${GLSLANG:-glslangValidator}
SPV=$HERE/06-trig-invariance.spv
DUMP_DIR=$HERE/06-trig-invariance.dump
GOLDEN=$HERE/06-trig-invariance.isa.golden
MODE=check
for arg in "$@"; do
  case $arg in
    --capture) MODE=capture ;;
    --check) MODE=check ;;
    -h|--help) sed -n 2,18p $0; exit 0 ;;
    *) echo "usage: $0 [--capture|--check]" >&2; exit 4 ;;
  esac
done

[ -x "$TRIG_DUMP" ] || { echo "error: trig_dump not built at $TRIG_DUMP" >&2; exit 3; }
"$GLSLANG" -V "$HERE/06-trig-invariance.comp" -o "$SPV"
mkdir -p "$DUMP_DIR"
TRIG_DUMP_DIR="$DUMP_DIR" "$TRIG_DUMP" "$SPV" > "$DUMP_DIR/stdout.txt" 2>&1

case $MODE in
  capture)
    if [ -f "$GOLDEN" ]; then
      echo "error: golden already exists at $GOLDEN" >&2
      echo "       refusing to overwrite. Delete manually or move aside." >&2
      exit 5
    fi
    cp "$DUMP_DIR/isa.txt" "$GOLDEN"
    echo "GOLDEN CAPTURED: $GOLDEN"
    echo "Review the file, then commit it explicitly. The test will FAIL on the next run without --capture until the golden is in place."
    exit 0
    ;;
  check)
    if [ ! -f "$GOLDEN" ]; then
      echo "TRIG INVARIANCE FAIL: golden missing at $GOLDEN" >&2
      echo "       run --capture on the reference hardware-under-test to seed it." >&2
      exit 2
    fi
    if ! diff -u "$GOLDEN" "$DUMP_DIR/isa.txt" > "$DUMP_DIR/diff.txt"; then
      echo "TRIG INVARIANCE FAILED" >&2
      head -200 "$DUMP_DIR/diff.txt" >&2
      exit 1
    fi
    echo "TRIG INVARIANCE PASS"
    exit 0
    ;;
esac
exit 4

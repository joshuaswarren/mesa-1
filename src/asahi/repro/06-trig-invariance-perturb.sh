#!/usr/bin/env bash
# Copyright 2026 Joshua Warren
# SPDX-License-Identifier: MIT
# 06 - trig lowering PERTURBATION test
#
# Applies a guaranteed-perturbation sentinel to the precise-math file
# (src/asahi/compiler/agx_nir_lower_math.c). Restoration on trap.
#
# To prove the trig-invariance gate catches the perturbation:
#   1. Run this script; it prepends #define AGX_PERTURB_TRIG to the
#      precise-math file (or SKIPs if absent).
#   2. Rebuild Mesa + trig_dump against the perturbed tree.
#   3. Run 06-trig-invariance-test.sh --check; expect exit 1.
#   4. The trap restores the file on exit; verify via diff.

set -euo pipefail
HERE=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)
MESA=${MESA_SRC:-$(cd $HERE/../../.. && pwd)}
PRECISE_FILE=${MESA_PRECISE_MATH_FILE:-$MESA/src/asahi/compiler/agx_nir_lower_math.c}
BACKUP_DIR=$HERE/06-trig-invariance.perturb.bak
SENTINEL='
#define AGX_PERTURB_TRIG 1
/* perturbation sentinel: trig lowering short-circuits */
'
mkdir -p "$BACKUP_DIR"

if [ ! -f "$PRECISE_FILE" ]; then
  echo "SKIP: $PRECISE_FILE does not exist on this Mesa tree" >&2
  echo "       (precise-math handler absent, perturbation cannot be applied)" >&2
  exit 77
fi

cp "$PRECISE_FILE" "$BACKUP_DIR/agx_nir_lower_math.c.bak"
restore() { cp "$BACKUP_DIR/agx_nir_lower_math.c.bak" "$PRECISE_FILE"; echo restored >&2; }
trap restore EXIT INT TERM

TMP="$BACKUP_DIR/agx_nir_lower_math.c.perturbed"
printf '$SENTINEL$' > "$TMP"
cat "$PRECISE_FILE" >> "$TMP"
cat "$TMP" > "$PRECISE_FILE"
echo "PERTURBATION INSTALLED at $PRECISE_FILE"
echo "To prove the gate catches this, rebuild Mesa + trig_dump and run:"
echo "  $HERE/06-trig-invariance-test.sh --check"
echo "The script will exit 1 on isa mismatch."
exit 0

#!/usr/bin/env bash
# 06 - trig lowering PERTURBATION test (semantic)
# Patches lower_fdiv to return false immediately (pass becomes a no-op for
# fdiv/frcp -> hardware path -> different isa.txt). Modes:
#   (default)  patch, print instructions, restore on exit
#   --hold     patch and LEAVE IT APPLIED (caller must rebuild + check + --restore)
#   --restore  restore original source
set -euo pipefail
HERE=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)
MESA=${MESA_SRC:-$(cd $HERE/../../.. && pwd)}
PRECISE_FILE=${MESA_PRECISE_MATH_FILE:-$MESA/src/asahi/compiler/agx_nir_lower_math.c}
BACKUP_DIR=$HERE/06-trig-invariance.perturb.bak
PATCHED=$BACKUP_DIR/agx_nir_lower_math.c.perturbed
mkdir -p "$BACKUP_DIR"
if [ ! -f "$PRECISE_FILE" ]; then echo SKIP: no $PRECISE_FILE >&2; exit 77; fi
case "${1:-}" in
  --restore)
    cp $BACKUP_DIR/agx_nir_lower_math.c.bak $PRECISE_FILE
    echo restored
    exit 0 ;;
  --hold) HOLD=1 ;;
  "") HOLD=0 ;;
  *) echo "usage: $0 [--hold|--restore]" >&2; exit 4 ;;
esac
cp $PRECISE_FILE $BACKUP_DIR/agx_nir_lower_math.c.bak
if [ ! -f "$PATCHED" ]; then
  python3 - "$PRECISE_FILE" "$PATCHED" <<"PYEOF"
import sys, pathlib
lines = pathlib.Path(sys.argv[1]).read_text().splitlines()
header_idx = next(i for i,l in enumerate(lines) if l.lstrip().startswith("lower_fdiv(nir_builder"))
start = next(i for i in range(header_idx, len(lines)) if lines[i].strip() == "{")
depth = 0
end = None
for i in range(start, len(lines)):
    depth += lines[i].count("{") - lines[i].count("}")
    if depth == 0:
        end = i
        break
indent = lines[header_idx][:len(lines[header_idx]) - len(lines[header_idx].lstrip())]
body = lines[start+1:end]
body2 = body[:]
for i,l in enumerate(body):
    if "res = div_rn(b, a, d);" in l:
        body2[i] = l.replace("div_rn(b, a, d)", "nir_fmul(b, a, nir_frcp(b, d)) /* perturbed: precise->fast fdiv */")
        break
patched = lines[:start+1] + body2 + lines[end:]
pathlib.Path(sys.argv[2]).write_text(chr(10).join(patched) + chr(10))
print("patched", end - start - 1, "body lines: early return false")
PYEOF
fi
cp $PATCHED $PRECISE_FILE
echo PERTURBATION INSTALLED: lower_fdiv returns false immediately
if [ "$HOLD" = 1 ]; then
  echo "HOLD: source left patched. Rebuild, run 06-trig-invariance-test.sh --check, expect exit 1."
  echo "Then run: $0 --restore"
else
  cp $BACKUP_DIR/agx_nir_lower_math.c.bak $PRECISE_FILE
  echo restored
fi

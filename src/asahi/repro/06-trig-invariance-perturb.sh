#!/usr/bin/env bash
# 06 - trig lowering PERTURBATION test (semantic)
# Wraps the lower_fdiv body in #ifdef AGX_PERTURB_TRIG so the pass is
# a no-op for fdiv/frcp -> hardware path -> different isa.txt
set -euo pipefail
HERE=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)
MESA=${MESA_SRC:-$(cd $HERE/../../.. && pwd)}
PRECISE_FILE=${MESA_PRECISE_MATH_FILE:-$MESA/src/asahi/compiler/agx_nir_lower_math.c}
BACKUP_DIR=$HERE/06-trig-invariance.perturb.bak
PATCHED=$BACKUP_DIR/agx_nir_lower_math.c.perturbed
mkdir -p "$BACKUP_DIR"

if [ ! -f "$PRECISE_FILE" ]; then
  echo SKIP: $PRECISE_FILE does not exist on this Mesa tree >&2
  exit 77
fi

cp $PRECISE_FILE $BACKUP_DIR/agx_nir_lower_math.c.bak
restore() { cp $BACKUP_DIR/agx_nir_lower_math.c.bak $PRECISE_FILE; echo restored >&2; }
trap restore EXIT INT TERM

python3 - $PRECISE_FILE $PATCHED <<INNEREOF
import sys, pathlib
src = pathlib.Path(sys.argv[1]).read_text()
lines = src.split(chr(10))
start = end = header_idx = None
for i, line in enumerate(lines):
    if start is None and line.lstrip().startswith('lower_fdiv(nir_builder'):
        header_idx = i
        for j in range(i, min(i + 3, len(lines))):
            if chr(123) in lines[j]:
                start = j + 1
                break
    if start is not None and end is None and line.strip() == chr(125) and i > start:
        end = i
        break
if start is None or end is None or header_idx is None:
    sys.exit('cannot find lower_fdiv body')
indent = lines[header_idx][:lines[header_idx].index('lower_fdiv')]
prefix = lines[:start]
suffix = lines[end + 1:]
body = lines[start:end]
patched = prefix + [
    indent + chr(35) + 'ifdef AGX_PERTURB_TRIG,',
    indent + chr(32) + chr(32) + chr(32) + chr(47) + chr(42) + chr(32) + chr(39) + 'pass is a no-op for fdiv/frcp: hardware path' + chr(39) + chr(32) + chr(42) + chr(47) + ',',
    indent + chr(32) + chr(32) + chr(32) + chr(39) + 'return false;' + chr(39) + ',',
    indent + chr(35) + 'else,',
] + body + [
    indent + chr(35) + 'endif,',
] + suffix
pathlib.Path(sys.argv[2]).write_text(chr(10).join(patched))
print('patched', end - start, 'lines of lower_fdiv body')
INNEREOF

cat $PATCHED > $PRECISE_FILE
echo PERTURBATION INSTALLED: lower_fdiv body wrapped in #ifdef AGX_PERTURB_TRIG
echo To prove the gate catches this, rebuild Mesa + trig_dump and run:
echo   $HERE/06-trig-invariance-test.sh --check
echo 'Expected: exit 1 (AGX isa differs from golden because fdiv is no longer lowered).'
exit 0
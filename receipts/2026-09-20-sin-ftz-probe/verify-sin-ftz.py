#!/usr/bin/env python3
"""Independent 64-word assert over the collected v3 window outputs.

Contract under test (hk/fdiv-fckey 095cb7e1b0a keyed emission):
  - default arm, every driver: sin() zero-compare select returns the argument
    for zero-comparing inputs (subnormals leak, zeros pass through)
  - FTZ arm, candidate driver: zero-comparing subnormals return the
    sign-preserving flushed zero
  - FTZ arm, e167 baseline: identical to default (pre-fix behavior)
  - every non-zero-compare row: bit-identical across all four legs
Exit 0 only when all rows hold.
"""
import struct, sys

def words(path):
    with open(path, "rb") as f:
        data = f.read()
    assert len(data) == 256, f"{path}: {len(data)} bytes != 256"
    return list(struct.unpack("<64I", data))

SUBNORMALS = {0x00000001, 0x007fffff, 0x80000001, 0x807fffff}
ZEROS = {0x00000000, 0x80000000}

legs = {}
for driver in ("e167", "fckey"):
    inp = words(f"window-v3/in-{driver}.bin")
    legs[(driver, "default")] = words(f"window-v3/out-{driver}-default.bin")
    legs[(driver, "ftz")] = words(f"window-v3/out-{driver}-ftz.bin")

inputs = legs[("e167", "default")] and words("window-v3/in-e167.bin")
assert inputs == words("window-v3/in-fckey.bin"), "input sets differ between drivers"

fails = []
for i, x in enumerate(inputs):
    d_e, f_e = legs[("e167", "default")][i], legs[("e167", "ftz")][i]
    d_c, f_c = legs[("fckey", "default")][i], legs[("fckey", "ftz")][i]

    if d_e != d_c:
        fails.append((i, hex(x), "default arms differ across drivers", d_e, d_c))
    if x in SUBNORMALS:
        if f_e != x:
            fails.append((i, hex(x), "baseline FTZ must leak the subnormal", f_e, x))
        want = x & 0x80000000
        if f_c != want:
            fails.append((i, hex(x), "candidate FTZ must return sign-preserving zero", f_c, want))
    elif x in ZEROS:
        if not (d_e == f_e == d_c == f_c == x):
            fails.append((i, hex(x), "zeros must pass through identically", d_e, f_e, d_c, f_c))
    else:
        if not (d_e == f_e == d_c == f_c):
            fails.append((i, hex(x), "non-zero-compare rows must be identical in all four legs", d_e, f_e, d_c, f_c))

for f in fails:
    print("FAIL:", f)
print(f"{64} words per leg x 4 legs = {64 * 4} words total; {len(fails)} failures")
sys.exit(1 if fails else 0)

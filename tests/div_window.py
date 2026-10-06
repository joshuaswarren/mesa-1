#!/usr/bin/env python3
"""Usage: div_window.py ICD COMPUTE_RUNNER [REF_ICD] [CHUNKS]; glslangValidator and numpy are required.

Default-path division and reciprocal against the correctly rounded (with FTZ)
quotient, bitwise, specials included: every fp32 reciprocal (64 chunks of
2^26) and the same number of hashed quotients, a quarter of them raw bits and
the rest with exponents at the zero, denormal, inf and 2^126 edges and with
quotients straddling 2^-126 and 2^128. With REF_ICD, every result must also
match that driver bitwise (for example the build before a lowering change).
CHUNKS is N (the first N of 64) or FIRST-LAST, so an interrupted sweep can
resume.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

args = sys.argv[1:]
icd, runner = map(os.path.abspath, args[:2])
ref = os.path.abspath(args[2]) if len(args) > 2 else None
span = args[3].split("-") if len(args) > 3 else ["64"]
chunks = range(int(span[0]), int(span[1]) + 1) if len(span) == 2 else range(int(span[0]))
here = Path(__file__).resolve().parent
N = 1 << 26
SEED = 20261006

def pcg(v):
    s = v * np.uint32(747796405) + np.uint32(2891336453)
    w = ((s >> ((s >> np.uint32(28)) + np.uint32(4))) ^ s) * np.uint32(277803737)
    return (w >> np.uint32(22)) ^ w

EA = np.array([0, 1, 2, 46, 47, 48, 49, 50, 126, 127, 128, 252, 253, 254, 255, 255], dtype=np.uint32)
ED = np.array([0, 0, 1, 2, 3, 126, 127, 128, 250, 251, 252, 253, 254, 255, 1, 252], dtype=np.uint32)

def operands(base):
    x = np.arange(N, dtype=np.uint32) + np.uint32(base)
    h0 = pcg(x ^ np.uint32(SEED))
    h1 = pcg(h0 + np.uint32(1))
    h2 = pcg(h1 + np.uint32(2))
    a, d, mode = h0.copy(), h1.copy(), h2 & 3
    exp = lambda bits, e: (bits & np.uint32(0x807fffff)) | (e.astype(np.uint32) << np.uint32(23))
    m = mode == 1
    a[m] = exp(a[m], EA[(h2[m] >> 2) & 15])
    d[m] = exp(d[m], ED[(h2[m] >> 6) & 15])
    m = mode == 2
    ed = 1 + (h2[m] >> 2) % 254
    e = ((h2[m] >> 10) % 14).astype(np.int64) + np.where(h2[m] & 0x80000000, 117, -128)
    a[m] = exp(a[m], np.clip(ed.astype(np.int64) + e, 0, 255))
    d[m] = exp(d[m], ed)
    m = mode == 3
    a[m] = exp(a[m], 1 + (h2[m] >> 2) % 254)
    d[m] = exp(d[m], 1 + (h2[m] >> 10) % 254)
    return a, d, x

def quotient(a, b):
    a = a.copy()
    b = b.copy()
    a[(a & 0x7fffffff) < 0x800000] &= 0x80000000
    b[(b & 0x7fffffff) < 0x800000] &= 0x80000000
    with np.errstate(all="ignore"):
        exact = a.view(np.float32).astype(np.float64) / b.view(np.float32).astype(np.float64)
        rounded = exact.astype(np.float32).view(np.uint32)
    flushed = np.abs(exact) < 2.0**-126
    rounded[flushed] = (a[flushed] ^ b[flushed]) & 0x80000000
    return rounded

def bad(got, want):
    both_nan = ((got & 0x7fffffff) > 0x7f800000) & ((want & 0x7fffffff) > 0x7f800000)
    return np.flatnonzero((got != want) & ~both_nan)

def run(spv, driver, base, path):
    path.with_suffix(".in").write_bytes(np.array([base, SEED], dtype=np.uint32).tobytes())
    env = dict(os.environ, VK_DRIVER_FILES=driver, VK_ICD_FILENAMES=driver, MESA_SHADER_CACHE_DISABLE="true")
    subprocess.run(["flock", "-w", "1800", "/tmp/m1-gpu.lock", runner, str(spv), str(path.with_suffix(".in")),
                    str(path), str(N * 8), str(N // 32 // 64)], env=env, check=True, timeout=2100)
    return np.fromfile(path, dtype=np.uint32).reshape(N, 2)

totals = {"div_oracle": 0, "recip_oracle": 0, "div_ref": 0, "recip_ref": 0}
examples = []
with tempfile.TemporaryDirectory(prefix="mesa-divwin-", dir="/dev/shm" if os.path.isdir("/dev/shm") else None) as td:
    td = Path(td)
    spv = td / "probe.spv"
    subprocess.run(["glslangValidator", "-V", str(here / "div_window.comp"), "-o", str(spv)], check=True, timeout=30)
    for c in chunks:
        base = c * N
        out = run(spv, icd, base, td / "out.bin")
        a, d, x = operands(base)
        checks = {"div_oracle": bad(out[:, 0], quotient(a, d)),
                  "recip_oracle": bad(out[:, 1], quotient(np.full(N, 0x3f800000, dtype=np.uint32), x))}
        if ref:
            r = run(spv, ref, base, td / "ref.bin")
            checks["div_ref"] = np.flatnonzero(out[:, 0] != r[:, 0])
            checks["recip_ref"] = np.flatnonzero(out[:, 1] != r[:, 1])
        for k, v in checks.items():
            totals[k] += len(v)
            for i in v[:2]:
                if len(examples) < 10:
                    examples.append({"check": k, "a": hex(int(a[i])), "d": hex(int(d[i])), "x": hex(int(x[i])),
                                     "got": [hex(int(w)) for w in out[i]]})
        print(f"chunk {c}: " + " ".join(f"{k}={len(v)}" for k, v in checks.items()), flush=True)

result = {"pairs": len(chunks) * N, "reciprocals": len(chunks) * N, "chunks": [chunks[0], chunks[-1]],
          "mismatches": totals if ref else
          {k: v for k, v in totals.items() if k.endswith("oracle")}, "examples": examples}
print(json.dumps(result, sort_keys=True))
sys.exit(any(totals.values()))

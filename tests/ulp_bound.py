#!/usr/bin/env python3
"""Usage: ulp_bound.py ICD COMPUTE_RUNNER; glslangValidator and numpy are required.

Default-path division contract: every result within 2.5 ULP of the correctly
rounded (with FTZ) quotient, the OpFDiv budget of the Vulkan SPIR-V
environment. Two sets: the exact 262,144-pair precise_math.py regression
input (same seed and construction) and a 66,048-pair boundary set straddling
the 2^-126 normal limit (the M2 Pro failure class, Basecamp 10277024959).
Specials (+-0, inf, NaN, flushed results) must match the oracle bitwise.
The correctly rounded contract itself lives in precise_math.py (nir_fp_exact
path); this test only pins the default.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

icd, runner = map(os.path.abspath, sys.argv[1:])
here = Path(__file__).resolve().parent

# Regression set: identical construction to precise_math.py (seed 20260913).
rng = np.random.default_rng(20260913)
n = 262144
inp = rng.integers(0, 2**32, (n, 4), dtype=np.uint32)
edges = np.array([0, 0x80000000, 1, 0x7fffff, 0x800000, 0x800001,
                  0x80800000, 0x3f800000, 0x3f7fffff, 0x3f800001,
                  0x40000000, 0x7f7fffff, 0xff7fffff, 0x7f800000,
                  0xff800000, 0x7fc00000], dtype=np.uint32)
inp[:256, 0] = np.repeat(edges, len(edges))
inp[:256, 1] = np.tile(edges, len(edges))
inp[256:65536, 0] = (inp[256:65536, 0] & 0x807fffff) | (rng.integers(1, 40, 65280, dtype=np.uint32) << 23)
inp[256:65536, 1] = (inp[256:65536, 1] & 0x807fffff) | (rng.integers(80, 160, 65280, dtype=np.uint32) << 23)
inp[:, 2] = 0x3f317218
inp[:, 3] = rng.integers(0x00800000, 0x7f800000, n, dtype=np.uint32)
inp[:7, 3] = [0x3f800000, 0x40000000, 0x40400000, 0x3f7fffff, 0x3f800001, 0x00800000, 0x7f7fffff]

# Boundary set: normal denominators in [1, 2^18), quotients
# (1 + j*2^-24)*2^-126 straddling the normal limit, j in [-64, 64],
# four sign combinations. a = RN(q * d) in fp32.
rngb = np.random.default_rng(20260913)
nb = 66048
db = ((rngb.integers(127, 145, nb, dtype=np.uint32) << 23) |
      rngb.integers(0, 1 << 23, nb, dtype=np.uint32))
j = rngb.integers(-64, 65, nb).astype(np.float64)
q64 = (1.0 + j * 2.0**-24) * 2.0**-126
ab = (q64 * db.view(np.float32).astype(np.float64)).astype(np.float32).view(np.uint32)
ab ^= rngb.integers(0, 2, nb, dtype=np.uint32) << 31
db ^= rngb.integers(0, 2, nb, dtype=np.uint32) << 31
bnd = np.stack([ab, db, np.full(nb, 0x3f317218, dtype=np.uint32),
                np.full(nb, 0x3f800000, dtype=np.uint32)], axis=1)
inp = np.concatenate([inp, bnd])
ntot = len(inp)

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

with tempfile.TemporaryDirectory(prefix="mesa-ulp-") as td:
    td = Path(td)
    inp.tofile(td / "in.bin")
    subprocess.run(["glslangValidator", "-V", str(here / "precise_math.comp"),
                    "-o", str(td / "probe.spv")], check=True)
    env = dict(os.environ, VK_DRIVER_FILES=icd, VK_ICD_FILENAMES=icd,
               MESA_SHADER_CACHE_DISABLE="true")
    subprocess.run(["flock", "-w", "30", "/tmp/m1-gpu.lock", runner,
                    str(td / "probe.spv"), str(td / "in.bin"), str(td / "out.bin"),
                    str(ntot * 6 * 4), str(ntot // 64)], env=env, check=True,
                   timeout=120)
    out = np.fromfile(td / "out.bin", dtype=np.uint32).reshape(ntot, 6)

def check(col, a_bits, b_bits):
    got = out[:, col]
    want = quotient(a_bits, b_bits)
    gf = got.copy().view(np.float32).astype(np.float64)
    wf = want.copy().view(np.float32).astype(np.float64)
    both_nan = ((got & 0x7fffffff) > 0x7f800000) & ((want & 0x7fffffff) > 0x7f800000)
    special = ~np.isfinite(wf) | (wf == 0.0)
    spec_bad = np.flatnonzero(special & (got != want) & ~both_nan)
    normal = np.flatnonzero(~special)
    with np.errstate(all="ignore"):
        dist = np.abs(gf[normal] - wf[normal]) / np.spacing(want[normal].view(np.float32).astype(np.float64))
    over = normal[dist > 2.5]
    return {"max_ulp": float(dist.max()) if len(normal) else 0.0,
            "over_2p5": len(over),
            "special_mismatch": len(spec_bad),
            "examples": [[hex(int(v)) for v in inp[i][:2]] for i in over[:5]][:5]}

result = {"pairs": ntot,
          "div": check(0, inp[:, 0], inp[:, 1]),
          "recip": check(1, np.full(ntot, 0x3f800000, dtype=np.uint32), inp[:, 1])}
print(json.dumps(result, sort_keys=True))
sys.exit(any(r["over_2p5"] or r["special_mismatch"] for r in result.values() if isinstance(r, dict)))

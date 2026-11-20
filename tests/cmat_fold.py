#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Usage: cmat_fold.py ICD COMPUTE_RUNNER [exact|time] [SEED]; glslc is required.

exact: run every form and data class with AGX_SIMDMAT_FOLD=0 and 1 and compare
the outputs bit for bit, from SSBO and shared memory, at 1 and 8 chained MACs.
The on arm must drop every fadd convert, so a pass that did not fire fails. The sparse class also checks both arms against a CPU product within
ULP_BOUND. NaN payload differences are counted apart from value differences.

time: paired, order-alternated fold off/on timings of a 16-step shader, SSBO and shared
memory sources, on synthetic normal fp16 data.

All data comes from a seeded PRNG. Prints one JSON line per case. Exit 0 means
no value mismatch in any case.
"""
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import struct
import subprocess
import sys
import tempfile

args = sys.argv[1:]
icd, runner = map(os.path.abspath, args[:2])
mode = args[2] if len(args) > 2 else "exact"
seed = int(args[3]) if len(args) > 3 else 20261030
here = Path(__file__).resolve().parent
TILES = 2048
ULP_BOUND = 1  # the matrix unit is not correctly rounded; see the sparse class

# Fragment layout of one 8x8 tile: lane -> (row, col of element 0).
ROW = lambda l: ((l >> 4) & 1) * 4 + ((l >> 1) & 3)
COL = lambda l: ((l >> 3) & 1) * 4 + (l & 1) * 2

def f32(x):
    return struct.pack("<f", x)

def h16_subnormal(r):
    return r.randrange(1, 0x400) | (r.getrandbits(1) << 15)

def h16_normal(r):
    return (r.randrange(1, 31) << 10) | r.randrange(0x400) | (r.getrandbits(1) << 15)

SPECIALS = [0x0000, 0x8000, 0x7c00, 0xfc00, 0x7e00, 0x7c01, 0xfe55, 0x7bff, 0xfbff, 0x0400, 0x0001, 0x8001, 0x03ff]

def h16(cls, r):
    if cls == "normal":
        return h16_normal(r)
    if cls == "subnormal":
        return h16_subnormal(r)
    if cls == "specials":
        return r.choice(SPECIALS)
    return r.choice([h16_normal, h16_subnormal, lambda q: q.choice(SPECIALS)])(r)

def word32(cls, r):
    """Large magnitudes put a subnormal fp16 product well into fp32 range."""
    return math.ldexp(1 + r.random(), r.randrange(10, 21) if cls == "subnormal" else r.randrange(-6, 7)) * r.choice((1, -1))

def h16_value(h):
    return struct.unpack("<e", struct.pack("<H", h))[0]

def dense(cls, r, steps):
    b = bytearray()
    for _ in range(TILES // steps * steps):
        b += b"".join(f32(word32(cls, r)) for _ in range(128))
        b += b"".join(struct.pack("<H", h16(cls, r)) for _ in range(128))
        b += b"".join(f32(word32("normal", r)) for _ in range(64))
    return bytes(b), None

def sparse(form, r):
    """One non-zero A[i][k] and B[k][j] per tile: D[i][j] is a*b exactly."""
    b, want = bytearray(), []
    for _ in range(TILES):
        i, j, k = (r.randrange(8) for _ in range(3))
        ah = h16_subnormal(r) if r.random() < 0.7 else h16_normal(r)
        bh = h16_subnormal(r) if r.random() < 0.7 else h16_normal(r)
        a32, b32 = word32("subnormal", r), word32("subnormal", r)
        A32, B32, A16, B16 = [0.0] * 64, [0.0] * 64, [0] * 64, [0] * 64
        for l in range(32):
            for e in range(2):
                n = 2 * l + e
                if ROW(l) == i and COL(l) + e == k:
                    A32[n], A16[n] = a32, ah
                if ROW(l) == k and COL(l) + e == j:
                    B32[n], B16[n] = b32, bh
        b += b"".join(map(f32, A32 + B32)) + struct.pack("<128H", *(A16 + B16)) + f32(0.0) * 64
        x = a32 if form == 0 else h16_value(ah)
        y = b32 if form == 1 else h16_value(bh)
        want.append((i, j, struct.unpack("<I", f32(x * y))[0]))
    return bytes(b), want

def compile_shader(tmp, form, steps, shared, wrap=0x7fffffff):
    spv = tmp / f"f{form}s{steps}m{shared}.spv"
    if not spv.exists():
        subprocess.run(["glslc", "--target-env=vulkan1.3", "-O", f"-DFORM={form}", f"-DSTEPS={steps}",
                        f"-DSHARED={shared}", f"-DWRAP={wrap}u", str(here / "cmat_fold.comp"), "-o", str(spv)], check=True)
    return spv

def converts(log):
    return len(re.findall(r"\bfadd\b", log))

def run(tmp, spv, data, fold, groups, reps=1, out_groups=None):
    i, o = tmp / "in.bin", tmp / "out.bin"
    i.write_bytes(data)
    env = dict(os.environ, VK_DRIVER_FILES=icd, MESA_SHADER_CACHE_DISABLE="true", AGX_SIMDMAT_FOLD=str(fold), AGX_MESA_DEBUG="shaders",
               COMPUTE_RUNNER_REPS=str(reps))
    p = subprocess.run([runner, str(spv), str(i), str(o), str((out_groups or groups) * 256), str(groups)], env=env,
                       capture_output=True, text=True, check=True)
    return o.read_bytes(), p.stdout + p.stderr

def is_nan(w):
    return (w & 0x7f800000) == 0x7f800000 and (w & 0x7fffff) != 0

def oracle(out, want):
    """Counts results that differ from the rounded product, and the largest
    difference in fp32 units in the last place."""
    w = struct.unpack(f"<{len(out) // 4}I", out)
    bad, ulp = 0, 0
    for t, (i, j, e) in enumerate(want):
        for l in range(32):
            for x in range(2):
                exp = e if ROW(l) == i and COL(l) + x == j else 0
                got = w[t * 64 + 2 * l + x]
                if got != exp:
                    bad += 1
                    ulp = max(ulp, abs(got - exp) if (got ^ exp) >> 31 == 0 else 1 << 31)
    return bad, ulp

def exact(tmp):
    failed = False
    for form in range(3):
        for steps, shared in ((1, 0), (1, 1), (8, 1)):
            spv = compile_shader(tmp, form, steps, shared)
            for cls in ("normal", "subnormal", "specials", "mixed", "sparse"):
                if cls == "sparse" and steps > 1:
                    continue
                r = random.Random(f"{seed}/{form}/{steps}/{shared}/{cls}")
                data, want = sparse(form, r) if cls == "sparse" else dense(cls, r, steps)
                groups = TILES // steps
                off, log_off = run(tmp, spv, data, 0, groups)
                on, log_on = run(tmp, spv, data, 1, groups)
                n = len(off) // 4
                a, b = struct.unpack(f"<{n}I", off), struct.unpack(f"<{n}I", on)
                diff = [(x, y) for x, y in zip(a, b) if x != y]
                nan = sum(is_nan(x) and is_nan(y) for x, y in diff)
                res = {"form": form, "steps": steps, "shared": shared, "class": cls, "elements": n,
                       "value_mismatches": len(diff) - nan, "nan_payload_diffs": nan,
                       "fadd_off": converts(log_off), "fadd_on": converts(log_on),
                       "first": [f"{x:08x}/{y:08x}" for x, y in diff if not (is_nan(x) and is_nan(y))][:3]}
                if want:
                    res["oracle_off"], res["oracle_on"] = oracle(off, want), oracle(on, want)
                    failed |= res["oracle_on"][1] > ULP_BOUND
                failed |= res["value_mismatches"] > 0 or res["fadd_on"] > 0 or res["fadd_off"] == 0
                print(json.dumps(res), flush=True)
    return failed

def timing(tmp):
    steps, groups, wrap = 16, 64 * 1024, 2048
    for shared in (0, 1):
        spv = compile_shader(tmp, 0, steps, shared, wrap)
        data, _ = dense("normal", random.Random(f"{seed}/time"), 1)
        data *= steps
        ms = {0: [], 1: []}
        for rep in range(16):
            for fold in (rep & 1, 1 - (rep & 1)):
                _, err = run(tmp, spv, data, fold, groups, reps=21, out_groups=wrap)
                ms[fold].append(float(err.rsplit("median_ms ", 1)[1].split()[0]))
        print(json.dumps({"shared": shared, "steps": steps, "groups": groups,
                          "median_off": statistics.median(ms[0]), "median_on": statistics.median(ms[1]),
                          "paired_delta_median": statistics.median(y - x for x, y in zip(ms[0], ms[1]))}), flush=True)
    return False

with tempfile.TemporaryDirectory() as t:
    sys.exit(1 if (timing if mode == "time" else exact)(Path(t)) else 0)

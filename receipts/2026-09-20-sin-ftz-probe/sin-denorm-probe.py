#!/usr/bin/env python3
"""sin() DenormFlushToZero keyed-emission probe.

Runs the two assembled modules (default vs DenormFlushToZero-declared) over
float bit patterns loaded unflushed from an SSBO, and dumps the raw output
bits. Baseline expectation below is RECORDED, not a device claim: the device
run happens in a Main-owned GPU window.

usage: sin-denorm-probe.py <icd.json> <tag> [--env K=V ...]
(reuses the precise-math mathrepro harness; build.sh assembles the .spv)

Locking: none here. The caller holds the host lock (read-only fd9 on M2) and
invokes this script inside it; MATHREPRO defaults to the native binary built
in the isolated build dir.
"""
import json, os, subprocess, sys
import numpy as np

icd, tag = sys.argv[1], sys.argv[2]
extra_env = dict(kv.split("=", 1) for kv in sys.argv[3:])
here = os.path.dirname(os.path.abspath(__file__))

NAMED = [
    (0x00000001, "pos_smallest_subnormal"),
    (0x007fffff, "pos_largest_subnormal"),
    (0x80000001, "neg_smallest_subnormal"),
    (0x807fffff, "neg_largest_subnormal"),
    (0x80000000, "neg_zero"),
    (0x00000000, "pos_zero"),
    (0x00800000, "flt_min"),
    (0x33800000, "norm_1e-8"),
    (0x3f800000, "one"),
    (0x3f317218, "ln2"),
    (0x7f800000, "pos_inf"),
    (0xff800000, "neg_inf"),
    (0x7fc00000, "qnan"),
]
n = 64  # one workgroup of 64, padded

bits = np.zeros(n, dtype=np.uint32)
names = {}
for i, (b, name) in enumerate(NAMED):
    bits[i] = b
    names[i] = name
inbin = os.path.join(here, f"in-{tag}.bin")
bits.tofile(inbin)

env = dict(os.environ, VK_ICD_FILENAMES=icd, **extra_env)

def run(spv):
    arm = spv.replace("sin-denorm-", "").replace("-v2", "").replace(".spv", "")
    outbin = inbin.replace("in-", f"out-{tag}-{arm}-")
    r = subprocess.run(
        [os.environ.get("MATHREPRO",
                        "/tmp/mesa-fckey-095cb/bin/mathrepro"),
         os.path.join(here, spv), inbin, outbin, str(n * 4), "1"],
        env=env, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        sys.exit(r.returncode)
    got = os.path.getsize(outbin)
    if got != n * 4:
        sys.exit(f"{arm}: output length {got} != expected {n * 4}")
    device_line = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ""
    out = np.fromfile(outbin, dtype=np.uint32)
    return out, device_line

out_default, dev_d = run("sin-denorm-default-v2.spv")
out_ftz, dev_f = run("sin-denorm-ftz-v2.spv")

result = {
    "tag": tag,
    "device_lines": {"default": dev_d, "ftz": dev_f},
    "inputs": {names.get(i, f"pad_{i}"): f"0x{b:08x}" for i, b in enumerate(bits) if i < len(NAMED)},
    "default_out": {names.get(i, f"pad_{i}"): f"0x{b:08x}" for i, b in enumerate(out_default) if i < len(NAMED)},
    "ftz_out": {names.get(i, f"pad_{i}"): f"0x{b:08x}" for i, b in enumerate(out_ftz) if i < len(NAMED)},
    "EXPECTED": {
        "note": "recorded CPU-side 2026-09-20 from the keyed-emission contract; "
                "NOT a device claim - device run pending in a Main-owned window",
        "subnormals_default": "out bits == input bits (argument leaks through the zero-compare select)",
        "subnormals_ftz": "out bits == input & 0x80000000 (sign-preserving flushed zero)",
        "pos_zero_neg_zero": "identical in both arms (select returns the argument/zero)",
        "normals_and_specials": "identical across arms (keying touches only the zero-compare path)",
    },
    "mathrepro": os.environ.get("MATHREPRO", "mathrepro"),
}
outjson = os.path.join(here, f"{tag}.json")
with open(outjson, "w") as f:
    json.dump(result, f, indent=1)
print(json.dumps(result, indent=1))

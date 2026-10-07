#!/usr/bin/env python3
"""Usage: large_constants.py ICD COMPUTE_RUNNER; glslangValidator and numpy are required.

llama.cpp IQ grid pattern: a 1024-entry constant table copied into shared memory with a lane-dependent index,
a barrier, then 96 lane-dependent lookups per invocation (4096 workgroups of 32). Runs the shader with
HK_LARGE_CONSTANTS unset and =1 (shader cache off, AGX_MESA_DEBUG=shaderdb). Both runs must match the CPU
model bitwise; with the flag, the shader must use no scratch (without it, Honeykrisp gives every invocation
a 4096 B scratch copy of the table).

Expected output (G13C, mesa-1 agent/hk-large-constants):
  off: 0 mismatches of 131072, scratch 4096
  on:  0 mismatches of 131072, scratch 0
"""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import numpy as np

icd, runner = map(os.path.abspath, sys.argv[1:3])
N_WG, WG, LOOKUPS, SEED = 4096, 32, 96, 20261007
table = [((i * 2654435761) >> 7) & 0xffffffff for i in range(1024)]
shader = f"""#version 450
layout(local_size_x = {WG}) in;
layout(std430, binding = 0) readonly buffer In {{ uint seed; }} p;
layout(std430, binding = 1) writeonly buffer Out {{ uint o[]; }} outp;
const uint T[1024] = uint[]({", ".join(f"{v}u" for v in table)});
shared uint S[1024];
void main()
{{
  uint lid = gl_LocalInvocationIndex;
  for (uint i = 0u; i < 1024u; i += {WG}u)
    S[i + lid] = T[i + lid];
  barrier();
  uint g = gl_GlobalInvocationID.x, acc = 0u;
  for (uint k = 0u; k < {LOOKUPS}u; k++)
    acc = acc * 31u + S[(g * 7u + k * 13u + p.seed) & 1023u];
  outp.o[g] = acc;
}}
"""

n = N_WG * WG
g = np.arange(n, dtype=np.uint64)
t = np.array(table, dtype=np.uint64)
acc = np.zeros(n, dtype=np.uint64)
for k in range(LOOKUPS):
    acc = (acc * 31 + t[(g * 7 + k * 13 + SEED) & 1023]) & 0xffffffff
want = acc.astype(np.uint32)

failed = False
with tempfile.TemporaryDirectory(prefix="large-constants-") as td:
    td = Path(td)
    (td / "p.comp").write_text(shader)
    subprocess.run(["glslangValidator", "-V", str(td / "p.comp"), "-o", str(td / "p.spv")], check=True,
                   capture_output=True, timeout=60)
    (td / "in.bin").write_bytes(np.array([SEED], dtype=np.uint32).tobytes())
    for arm, flag in (("off", None), ("on", "1")):
        env = dict(os.environ, VK_DRIVER_FILES=icd, VK_ICD_FILENAMES=icd, AGX_MESA_DEBUG="shaderdb",
                   MESA_SHADER_CACHE_DISABLE="true")
        env.pop("HK_LARGE_CONSTANTS", None)
        if flag:
            env["HK_LARGE_CONSTANTS"] = flag
        r = subprocess.run(["flock", "-w", "1800", "/tmp/m1-gpu.lock", runner, str(td / "p.spv"),
                            str(td / "in.bin"), str(td / "out.bin"), str(n * 4), str(N_WG)],
                           env=env, capture_output=True, text=True, timeout=2100)
        stats = [l for l in (r.stdout + r.stderr).splitlines() if l.startswith("CS shader")]
        scratch = [int(m.group(1)) for l in stats for m in [re.search(r"(\d+) scratch", l)] if m]
        bad = int((np.fromfile(td / "out.bin", dtype=np.uint32) != want).sum())
        print(f"{arm}: {bad} mismatches of {n}, scratch {scratch[0] if scratch else 'unknown'}")
        failed |= r.returncode != 0 or bad != 0 or (flag and scratch != [0])
sys.exit(1 if failed else 0)

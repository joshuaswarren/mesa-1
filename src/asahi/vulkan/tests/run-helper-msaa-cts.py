#!/usr/bin/env python3
# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT
"""Repeat the G17P helper-texture and MSAA builtin regressions with Vulkan CTS.

Run under the hardware session's queue, journal and thermal guard. Each
iteration creates a fresh CTS process and disables Mesa's shader cache.
"""
import argparse
import collections
import json
import os
from pathlib import Path
import re
import subprocess

CASES = [
    "dEQP-VK.glsl.helper_invocations.load_from_texture",
    "dEQP-VK.pipeline.monolithic.multisample_shader_builtin.sample_id.128_128_1.samples_4",
    "dEQP-VK.pipeline.monolithic.multisample_shader_builtin.sample_id.137_191_1.samples_2",
    "dEQP-VK.pipeline.monolithic.multisample_shader_builtin.sample_position.distribution.128_128_1.samples_2",
    "dEQP-VK.pipeline.monolithic.multisample_shader_builtin.sample_position.distribution.128_128_1.samples_4",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deqp", type=Path, required=True)
    parser.add_argument("--icd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=45)
    args = parser.parse_args()
    if args.iterations <= 0 or args.timeout <= 0:
        parser.error("iterations and timeout must be positive")
    deqp, icd, output = (p.resolve() for p in (args.deqp, args.icd, args.output))
    output.mkdir(parents=True, exist_ok=False)
    caselist = output / "cases.txt"
    caselist.write_text("\n".join(CASES) + "\n")
    env = dict(os.environ, VK_DRIVER_FILES=str(icd), MESA_SHADER_CACHE_DISABLE="true")
    for var in ("DISPLAY", "WAYLAND_DISPLAY"):
        env.pop(var, None)
    counts = {case: collections.Counter() for case in CASES}
    completed = 0
    for iteration in range(1, args.iterations + 1):
        qpa = output / f"{iteration}.qpa"
        command = [str(deqp), f"--deqp-caselist-file={caselist}",
                   f"--deqp-log-filename={qpa}", "--deqp-log-images=disable",
                   "--deqp-log-shader-sources=disable"]
        with (output / f"{iteration}.log").open("w") as log:
            proc = subprocess.run(command, cwd=deqp.parent, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, timeout=args.timeout)
        results = re.findall(r'<TestCaseResult[^>]*CasePath="([^"]+)".*?'
                             r'<Result StatusCode="([^"]+)"', qpa.read_text(), re.S)
        if len(results) != len(CASES) or {case for case, _ in results} != set(CASES):
            raise RuntimeError(f"iteration {iteration}: incomplete CTS results")
        for case, status in results:
            counts[case][status] += 1
        completed += 1
        summary = {"completed_iterations": completed, "requested_iterations": args.iterations,
                   "counts": counts, "command": command, "icd": str(icd)}
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if proc.returncode not in (0, 1):
            raise RuntimeError(f"iteration {iteration}: CTS exit {proc.returncode}")
    success = all(counts[case] == {"Pass": args.iterations} for case in CASES)
    print(json.dumps({"completed_iterations": completed, "all_pass": success}))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())

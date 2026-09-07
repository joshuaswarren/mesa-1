#!/usr/bin/env python3
"""Run a Gallium test binary under the asahi no-op DRM shim.

The shim intercepts open() through LD_PRELOAD, so it has to be in the
environment of the process itself; meson's test env cannot name a build target
path portably from the test declaration alone. usage: run-shim.py <shim.so>
<binary> [args...]
"""
import os
import subprocess
import sys

shim, binary = sys.argv[1], sys.argv[2]
env = dict(os.environ)
preload = env.get("LD_PRELOAD")
env["LD_PRELOAD"] = shim + (":" + preload if preload else "")
env.setdefault("ASAHI_SHIM_CHIP", "g17p")
sys.exit(subprocess.call([binary] + sys.argv[3:], env=env))

#!/usr/bin/env python3
"""Run the public EGL/GLES resource oracle on Mesa's no-op Asahi DRM shim."""

import json
import os
import pathlib
import subprocess
import sys
import tempfile


shim, vendor, gallium, binary = map(pathlib.Path, sys.argv[1:5])
binary_args = sys.argv[5:]
env = dict(os.environ)

with tempfile.TemporaryDirectory(prefix="asahi-gl-shim-") as tmp:
    root = pathlib.Path(tmp)
    driver = root / "asahi_dri.so"
    driver.symlink_to(gallium.resolve())

    vendor_json = root / "50_mesa.json"
    vendor_json.write_text(json.dumps({
        "file_format_version": "1.0.0",
        "ICD": {"library_path": str(vendor.resolve())},
    }))

    preload = env.get("LD_PRELOAD")
    env["LD_PRELOAD"] = str(shim) + (":" + preload if preload else "")
    env["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(vendor_json)
    env["LIBGL_DRIVERS_PATH"] = str(root)
    env["GBM_BACKENDS_PATH"] = str(gallium.parents[3] / "gbm/backends/dri")
    env["MESA_LOADER_DRIVER_OVERRIDE"] = "asahi"
    env["EGL_PLATFORM"] = "surfaceless"
    env.setdefault("ASAHI_SHIM_CHIP", "g17p")
    env["ASAHI_GL_NOOP_SHIM"] = "1"
    env.setdefault("ASAHI_SHIM_VALIDATE_G17P_STREAM", "1")
    env.setdefault("ASAHI_GL_RENDER_NODE", "/dev/dri/renderD128")
    # The runner knows which Gallium it meant to test; say so, so an oracle
    # that checks provenance cannot silently grade a different driver.
    env.setdefault("ASAHI_GL_EXPECT_DRIVER", str(gallium.resolve()))

    # Every other shim oracle runs with MESA_SHADER_CACHE_DISABLE=true, so the
    # disk-cache HIT path -- which is what every launch after the first one
    # takes -- was never executed here at all.  ASAHI_GL_SHIM_WARM_CACHE runs
    # the client twice against one private cache directory and requires both
    # runs to pass, so a compiled-shader property that survives compilation but
    # not serialisation fails on a laptop instead of on the target.
    if env.pop("ASAHI_GL_SHIM_WARM_CACHE", None):
        cache = root / "shader-cache"
        cache.mkdir()
        env.pop("MESA_SHADER_CACHE_DISABLE", None)
        env["MESA_SHADER_CACHE_DIR"] = str(cache)

        for run in range(2):
            status = subprocess.call([str(binary), *binary_args], env=env)
            if status != 0:
                print(f"run-gl-shim: warm-cache run {run + 1} failed "
                      f"({status})", file=sys.stderr)
                sys.exit(status)
        sys.exit(0)

    sys.exit(subprocess.call([str(binary), *binary_args], env=env))

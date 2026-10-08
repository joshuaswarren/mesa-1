#!/usr/bin/env python3
"""Run an EGL/GLES test on Mesa's no-op Asahi DRM shim, through the asahi
Gallium driver or, with ASAHI_GL_SHIM_VULKAN=<libvulkan_asahi.so>, through
Zink on that Vulkan driver."""

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
    vulkan = env.pop("ASAHI_GL_SHIM_VULKAN", None)
    gallium_driver = "zink" if vulkan else "asahi"
    driver = root / f"{gallium_driver}_dri.so"
    driver.symlink_to(gallium.resolve())
    if vulkan:
        icd = root / "asahi_icd.json"
        icd.write_text(json.dumps({
            "file_format_version": "1.0.0",
            "ICD": {"library_path": vulkan, "api_version": "1.4.0"},
        }))
        env["VK_DRIVER_FILES"] = str(icd)

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
    env["MESA_LOADER_DRIVER_OVERRIDE"] = gallium_driver
    env["EGL_PLATFORM"] = "surfaceless"
    env.setdefault("ASAHI_SHIM_CHIP", "g17p")
    env["ASAHI_GL_NOOP_SHIM"] = "1"
    env.setdefault("ASAHI_SHIM_VALIDATE_G17P_STREAM", "1")

    sys.exit(subprocess.call([str(binary), *binary_args], env=env))

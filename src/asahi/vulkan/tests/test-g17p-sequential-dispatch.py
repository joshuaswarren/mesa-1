#!/usr/bin/env python3
# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT
"""Validate indirect record roles and preserve a single compute submission."""

import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def check_submission(text):
    submits = re.findall(r"^submit \d+ .*", text, re.MULTILINE)
    commands = re.findall(r"^  cmd (\S+) .*", text, re.MULTILINE)
    lengths = re.findall(r"^  cmd compute .* cdm_len=(\d+)$", text, re.MULTILINE)
    assert len(submits) == 1, f"expected one ioctl: {text}"
    assert commands.count("compute") == 1 and "render" not in commands, text
    # One direct application launch, one direct helper, one compact indirect
    # application launch, their existing flags=0 tails, and a stream terminator.
    assert lengths == ["120"], f"changed compute stream extent: {text}"


def main():
    build = Path(sys.argv[1]).resolve()
    vulkan = build / "src/asahi/vulkan"
    with tempfile.TemporaryDirectory(prefix="g17p-sequential-") as directory:
        log = Path(directory) / "submit.log"
        env = os.environ.copy()
        env.update(
            ASAHI_SHIM_CHIP="g17p",
            ASAHI_SHIM_VALIDATE_G17P_STREAM="1",
            ASAHI_SHIM_SUBMIT_LOG=str(log),
            LD_PRELOAD=str(build / "src/asahi/drm-shim/libasahi_noop_drm_shim.so"),
            MESA_SHADER_CACHE_DISABLE="true",
        )
        # Use the generated ICD regardless of the build host architecture.
        icds = list(vulkan.glob("asahi_devenv_icd.*.json"))
        assert len(icds) == 1, f"expected one development ICD: {icds}"
        env["VK_DRIVER_FILES"] = str(icds[0])
        result = subprocess.run(
            [str(vulkan / "g17p-vulkan-sequential-dispatch"), "--submit-only"],
            env=env, text=True, capture_output=True, timeout=20,
        )
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        assert result.returncode == 0, f"probe failed: {result.returncode}"
        text = log.read_text()
        check_submission(text)
        sys.stdout.write(text)
    print("PASS: helper/application roles verified; one ioctl, one 120-byte compute stream")


if __name__ == "__main__":
    main()

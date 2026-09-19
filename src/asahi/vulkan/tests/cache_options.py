#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run with VK_DRIVER_FILES pointing to the Honeykrisp ICD under test."""
import os
import re
import subprocess

OPTIONS = ("AGX_SIMDMAT", "AGX_DECODE_Q4", "AGX_HWMAT_VEC2")


def inspect(options):
    env = {k: v for k, v in os.environ.items() if k not in OPTIONS}
    env.update(options)
    result = subprocess.run(["vulkaninfo"], env=env, capture_output=True,
                            text=True, check=True, timeout=30)
    names = re.findall(r"^\s*deviceName\s*=\s*(.+)$", result.stdout, re.MULTILINE)
    assert len(names) == 1 and "Apple" in names[0], names
    ids = tuple(re.findall(r"^\s*(?:pipelineCacheUUID|shaderBinaryUUID)\s*=\s*(\S+)",
                           result.stdout, re.MULTILINE))
    assert len(ids) == 2, ids
    advertised = bool(re.search(r"^\s*VK_KHR_cooperative_matrix\s+:",
                                result.stdout, re.MULTILINE))
    print(options, ids, "cooperative_matrix=" + str(advertised), flush=True)
    return ids, advertised


def main():
    base, advertised = inspect({})
    assert advertised, "cooperative matrices must be on by default"
    off, advertised = inspect({"AGX_SIMDMAT": "0"})
    assert not advertised and all(a != b for a, b in zip(base, off))
    assert inspect({"AGX_SIMDMAT": "1"}) == (base, True), \
        "AGX_SIMDMAT=1 is the default and must not change the cache identity"
    seen = {base}
    for option in OPTIONS[1:]:
        changed, advertised = inspect({option: "1"})
        assert advertised and all(a != b for a, b in zip(base, changed)), option
        assert changed not in seen, option
        seen.add(changed)
        assert inspect({option: ""}) == (changed, True)
    assert inspect({}) == (base, True), "unsetting options must restore identity"
    print("PASS: default-on advertisement and cache identities track compiler options")


if __name__ == "__main__":
    main()

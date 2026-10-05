#!/usr/bin/env python3
# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT
"""Mark a PE file as a Wine builtin DLL.

Wine's loader only pairs a PE module with its unix library (aarch64-unix/*.so)
when the image carries the string "Wine builtin DLL" in the 32 bytes that
follow the 64-byte DOS header; winebuild --builtin writes the same bytes.
"""
import struct
import sys

SIGNATURE = b"Wine builtin DLL\0"


def main(path):
    with open(path, "r+b") as f:
        header = f.read(64)
        if len(header) != 64 or header[:2] != b"MZ":
            sys.exit(f"{path}: not a PE file")
        e_lfanew = struct.unpack_from("<I", header, 60)[0]
        if e_lfanew < 64 + len(SIGNATURE):
            sys.exit(f"{path}: no room for the Wine signature (e_lfanew=0x{e_lfanew:x})")
        f.seek(64)
        f.write(SIGNATURE)
    print(f"{path}: marked as Wine builtin")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)

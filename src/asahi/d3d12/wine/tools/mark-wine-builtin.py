#!/usr/bin/env python3
# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT
"""Mark a PE file as a Wine builtin DLL and give an ARM64EC image the
AMD64 machine type.

Wine's loader only pairs a PE module with its unix library (aarch64-unix/*.so)
when the image carries the string "Wine builtin DLL" in the 32 bytes that
follow the 64-byte DOS header; winebuild --builtin writes the same bytes.

lld writes IMAGE_FILE_MACHINE_ARM64EC (0xA641) into the COFF header of a pure
ARM64EC link. wineserver's is_machine_64bit()/is_machine_supported() only know
AMD64 and ARM64, so such an image is refused with STATUS_INVALID_IMAGE_FORMAT.
Microsoft's own ARM64EC DLLs carry IMAGE_FILE_MACHINE_AMD64 plus the CHPE
metadata in the load config, which is what both Windows and Wine
(server/mapping.c: is_hybrid = !!CHPEMetadataPointer) key off, so rewrite the
machine field the same way.
"""
import struct
import sys

SIGNATURE = b"Wine builtin DLL\0"
IMAGE_FILE_MACHINE_AMD64 = 0x8664
IMAGE_FILE_MACHINE_ARM64EC = 0xA641


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
        f.seek(e_lfanew)
        sig, machine = struct.unpack("<4sH", f.read(6))
        if sig != b"PE\0\0":
            sys.exit(f"{path}: bad PE signature at 0x{e_lfanew:x}")
        if machine == IMAGE_FILE_MACHINE_ARM64EC:
            f.seek(e_lfanew + 4)
            f.write(struct.pack("<H", IMAGE_FILE_MACHINE_AMD64))
            print(f"{path}: machine ARM64EC -> AMD64 (hybrid image)")
        elif machine != IMAGE_FILE_MACHINE_AMD64:
            sys.exit(f"{path}: unexpected machine 0x{machine:04x}")
        else:
            print(f"{path}: header machine 0x{machine:04x} (AMD64)")
    print(f"{path}: marked as Wine builtin")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)

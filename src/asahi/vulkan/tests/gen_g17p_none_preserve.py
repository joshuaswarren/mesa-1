#!/usr/bin/env python3
# Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT
import pathlib
import struct
import subprocess
import tempfile

root = pathlib.Path(__file__).resolve().parent
for stage in ("vert", "frag"):
    source = root / f"g17p_vulkan_none_preserve.{stage}"
    with tempfile.TemporaryDirectory() as tmp:
        binary = pathlib.Path(tmp) / "shader.spv"
        subprocess.run(["glslangValidator", "-V", "--target-env", "vulkan1.3",
                        str(source), "-o", str(binary)], check=True)
        subprocess.run(["spirv-val", "--target-env", "vulkan1.3", str(binary)],
                       check=True)
        data = binary.read_bytes()
    words = struct.unpack(f"<{len(data) // 4}I", data)
    text = ("/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT */\n"
            f"/* Generated from g17p_vulkan_none_preserve.{stage}. */\n"
            "#include <stdint.h>\n"
            f"static const uint32_t none_{stage}[] = {{")
    text += ",".join(hex(word) for word in words) + "};\n"
    (root / f"g17p_vulkan_none_preserve_{stage}.h").write_text(text)

#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Print ISG1/OSG1/PSG1 records without changing the shader container."""
import hashlib
import json
from pathlib import Path
import struct
import sys


def signatures(path):
    data = path.read_bytes()
    if len(data) < 32 or data[:4] != b"DXBC":
        raise ValueError(f"{path}: not a shader container")
    count = struct.unpack_from("<I", data, 28)[0]
    result = {"file": str(path), "sha256": hashlib.sha256(data).hexdigest(),
              "signatures": {}}
    for i in range(count):
        offset = struct.unpack_from("<I", data, 32 + 4 * i)[0]
        tag, size = struct.unpack_from("<4sI", data, offset)
        if offset + 8 + size > len(data):
            raise ValueError(f"{path}: truncated chunk")
        if tag not in (b"ISG1", b"OSG1", b"PSG1"):
            continue
        chunk = data[offset + 8:offset + 8 + size]
        elements, start = struct.unpack_from("<II", chunk)
        records = []
        for n in range(elements):
            stream, name, index, system, component, register, mask, used, _, precision = \
                struct.unpack_from("<6IBBHI", chunk, start + 32 * n)
            records.append({
                "semantic": chunk[name:chunk.index(b"\0", name)].decode("ascii"),
                "index": index, "stream": stream, "system_value": system,
                "component_type": component, "register": register,
                "mask": mask, "used_mask": used, "min_precision": precision,
            })
        result["signatures"][tag.decode("ascii")] = records
    return result


if __name__ == "__main__":
    print(json.dumps([signatures(Path(p)) for p in sys.argv[1:]], indent=2))

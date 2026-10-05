#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize separately versioned inputs; never rewrite the frozen reference."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('proton_tests', type=Path)
parser.add_argument('frozen_shaders', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
manifest = json.loads(Path(__file__).with_name('g6-legal-v1.json').read_text())
header = args.proton_tests / 'shaders/mesh_shader/headers/ps_green.h'
assert hashlib.sha256(header.read_bytes()).hexdigest() == manifest['upstream_header_sha256']
match = re.search(r'ps_green_code_dxil\[\]\s*=\s*\{(.*?)\};', header.read_text(), re.S)
assert match
ps = bytes(int(x, 16) for x in re.findall(r'0x([0-9a-fA-F]{2})\b', match[1]))
assert ps[:4] == b'DXBC' and struct.unpack_from('<I', ps, 24)[0] == len(ps)
chunks = {}
for i in range(struct.unpack_from('<I', ps, 28)[0]):
    off = struct.unpack_from('<I', ps, 32 + i * 4)[0]
    size = struct.unpack_from('<I', ps, off + 4)[0]
    assert off + 8 + size <= len(ps)
    chunks[ps[off:off + 4]] = ps[off + 8:off + 8 + size]
assert struct.unpack_from('<I', chunks[b'ISG1'])[0] == 0
inputs = {'ps-green.dxil': ps}
for name in ('gs_input_vs.dxil', 'gs_point.dxil'):
    inputs[name] = (args.frozen_shaders / name).read_bytes()
for name, data in inputs.items():
    assert hashlib.sha256(data).hexdigest() == manifest['sha256'][name]
args.output.mkdir(parents=True, exist_ok=False)
for name, data in inputs.items():
    (args.output / name).write_bytes(data)
shutil.copy2(Path(__file__).with_name('g6-legal-v1.json'), args.output / 'MANIFEST.json')
print(args.output)

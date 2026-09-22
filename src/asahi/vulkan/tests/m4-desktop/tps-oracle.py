#!/usr/bin/env python3
"""Generate exact SSBO oracles for AGX scratch lifetime and register spilling.

This prepares files only. Execute probe.spv with vk-buffer-oracle through
gpu-experiment.py on M4, adding AGX_MESA_DEBUG=spill for the spill case.
"""
import argparse
import hashlib
import json
import struct
import subprocess
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('case', choices=['repeat', 'spill'])
p.add_argument('--vectors', type=int, default=8, choices=range(2, 65))
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
n = 32 if a.case == 'repeat' else a.vectors
stride = n + 1 if a.case == 'repeat' else n * 2
body = [f'uint b=gl_GlobalInvocationID.x*{stride};']
if a.case == 'repeat':
    body += ['uvec4 a[32];']
    body += [f'a[{i}]=w[b+{i}];' for i in range(n)]
    body += ['uint index=w[b+32].x&31;',
             'w[b+32]=a[index]^a[0]^a[5]^a[31];']
else:
    body += [f'uvec4 v{i}=w[b+{i}];' for i in range(n)]
    body += [f'w[b+{n+i}]=v{i}^uvec4(0xa55aa55a);' for i in range(n)]
source = '''#version 450
layout(local_size_x=32) in;
layout(set=0,binding=0,std430) coherent volatile buffer B{uvec4 w[];};
void main(){
''' + '\n'.join(body) + '\n}\n'
(a.output / 'probe.comp').write_text(source)
initial, expected = [], []
for lane in range(32):
    values = [(0x90001234 + lane * 811 + i * 101) & 0xffffffff
              for i in range(n * 4)]
    if a.case == 'repeat':
        initial += values + [lane, 0, 0, 0]
        expected += values + [values[lane*4+c] ^ values[c] ^
                              values[20+c] ^ values[124+c] for c in range(4)]
    else:
        initial += values + [0] * len(values)
        expected += values + [v ^ 0xa55aa55a for v in values]
for name, words in [('input', initial), ('reference', expected)]:
    (a.output / (name + '.bin')).write_bytes(struct.pack('<'+'I'*len(words), *words))
command = ['glslangValidator', '-V', str(a.output/'probe.comp'),
           '-o', str(a.output/'probe.spv')]
compiled = subprocess.run(command, text=True, capture_output=True)
(a.output/'compile.log').write_text(compiled.stdout + compiled.stderr)
compiled.check_returncode()
manifest = {'case': a.case, 'vectors': n, 'words': len(initial),
            'workgroups': [1, 1, 1], 'command': command,
            'gpu_executed': False, 'sha256': {
                f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                for f in a.output.iterdir() if f.is_file()}}
(a.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps(manifest, indent=2))

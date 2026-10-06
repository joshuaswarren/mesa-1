#!/usr/bin/env python3
# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT
"""Build both frozen suites on a Linux reference machine, without changing inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

NAMES = ('g1_smoke g2_smoke g3_smoke g4_smoke g4_triangle g5_depth '
         'g5_descriptors g5_atomics g5_control_flow g5_wave g5_texture '
         'g6_geometry g7_tessellation g8_mesh g9_amplification '
         'g10_vertex_buffers g11_indexed_instanced').split()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--mesa', type=Path, required=True, help='libd3d12agx.so')
    parser.add_argument('--proton', type=Path, required=True, help='native Proton build/')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    libs = {'mesa': args.mesa.resolve(),
            'proton': args.proton.resolve() / 'libs/d3d12/libvkd3d-proton-d3d12.so'}
    metadata = {'libraries': {}, 'compiler': subprocess.check_output(
        ['cc', '--version'], text=True), 'commands': []}
    for driver, lib in libs.items():
        metadata['libraries'][driver] = {'path': str(lib),
            'sha256': hashlib.sha256(lib.read_bytes()).hexdigest()}
        for version in ('v1', 'v2'):
            binary_dir = out / driver / version / 'bin'
            binary_dir.mkdir(parents=True)
            for name in NAMES:
                cmd = ['cc', '-std=c11', '-D_GNU_SOURCE', '-O2', '-g', '-I/usr/include/wsl/stubs',
                       '-I/usr/include/directx']
                if driver == 'proton':
                    cmd += ['-include', str(here / 'proton-abi.h')]
                cmd += [str(here / version / (name + '.c')), '-o',
                        str(binary_dir / ('d3d12agx_' + name)), str(lib),
                        '/usr/lib64/libDirectX-Guids.a', '-lstdc++']
                metadata['commands'].append(cmd)
                stem = binary_dir.parent / (name + '.build')
                stem.with_suffix('.build-command').write_text(shlex.join(cmd) + '\n')
                with stem.with_suffix('.build-log').open('w') as log:
                    subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
                print(driver, version, name, 'built', flush=True)
    (out / 'build.json').write_text(json.dumps(metadata, indent=2) + '\n')
    artifacts = sorted(p for p in out.rglob('*') if p.is_file())
    (out / 'ARTIFACTS.sha256').write_text(''.join(
        hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + str(p.relative_to(out)) + '\n'
        for p in artifacts))


if __name__ == '__main__':
    main()

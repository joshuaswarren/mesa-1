#!/usr/bin/env python3
# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT
"""Rebuild v2's two shader corrections with an existing Mesa Linux build.

Reads build commands/static libraries; writes only to a new output directory.
The caller can compare outputs to committed v2 shaders before adopting them.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mesa_build', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    build = args.mesa_build.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent / 'v2/make_probe.c'
    entry = next(e for e in json.loads((build / 'compile_commands.json').read_text())
                 if e['file'].endswith('tests/make_probe.c'))
    compile_cmd = entry.get('arguments') or shlex.split(entry['command'])
    compile_cmd[compile_cmd.index('-o') + 1] = str(out / 'make-probe.o')
    compile_cmd[-1] = str(source)
    for flag in ('-MF', '-MQ', '-MT'):
        if flag in compile_cmd:
            i = compile_cmd.index(flag)
            del compile_cmd[i:i + 2]
    target = 'src/asahi/d3d12/d3d12agx_make_probe'
    link_cmd = shlex.split(subprocess.check_output(
        ['ninja', '-C', str(build), '-t', 'commands', target], text=True).splitlines()[-1])
    link_cmd[link_cmd.index('-o') + 1] = str(out / 'make-probe')
    index = next(i for i, s in enumerate(link_cmd) if s.endswith('tests_make_probe.c.o'))
    link_cmd[index] = str(out / 'make-probe.o')
    commands = [compile_cmd, link_cmd]
    commands += [[str(out / 'make-probe'), shader, str(out / (shader + '.dxil'))]
                 for shader in ('solid_link', 'tess_ds')]
    (out / 'COMMANDS.txt').write_text(''.join(shlex.join(c) + '\n' for c in commands))
    with (out / 'BUILD.log').open('w') as log:
        for cmd in commands:
            subprocess.run(cmd, cwd=build, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()

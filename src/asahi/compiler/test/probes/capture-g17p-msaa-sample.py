#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Capture register-sourced G17P MSAA sample staging without executing a GPU.

Usage: python3 capture-g17p-msaa-sample.py NEW_OUTPUT_DIRECTORY
Requires Apple's metal, metallib and metal-nt tools through xcrun.
Retains source, pipeline, native objects, section hashes and command logs.
"""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    'native_sections', HERE / 'capture-g17p-mrt-eot.py')
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)

VERTEX = '''
#include <metal_stdlib>
using namespace metal;
struct DepthOutput { float depth [[depth(any)]]; };
vertex float4 vertex_main(uint id [[vertex_id]]) {
    return float4(float2((id << 1) & 2, id & 2) * 2.0 - 1.0, 0.0, 1.0);
}
'''
PACK = bytes.fromhex('1705540400100000d0c00800')


def capture(output, name, expression):
    directory = output / name
    directory.mkdir()
    source = VERTEX + '''
fragment DepthOutput fragment_main(float4 position [[position]],
                                   depth2d_ms<float> source [[texture(0)]]) {
    return {source.read(uint2(position.xy), ''' + expression + ''')};
}
'''
    (directory / 'input.metal').write_text(source)
    pipeline = {'pipelines': {'render_pipelines': [{
        'vertex_function': 'vertex_main',
        'fragment_function': 'fragment_main',
        'depth_attachment_pixel_format': 'Depth32Float',
    }]}}
    (directory / 'pipeline.mtlp-json').write_text(json.dumps(pipeline, indent=2))
    commands = [
        ['xcrun', 'metal', '-c', str(directory / 'input.metal'),
         '-o', str(directory / 'input.air')],
        ['xcrun', 'metallib', str(directory / 'input.air'),
         '-o', str(directory / 'input.metallib')],
        ['xcrun', 'metal-nt', '-arch', 'applegpu_g17p',
         '-platform_version', 'macos', '27.0', '27.0',
         '-N', str(directory / 'pipeline.mtlp-json'),
         str(directory / 'input.metallib'), '-o', str(directory / 'output.nat')],
    ]
    receipt = {'commands': commands, 'hardware_execution': False}
    for index, command in enumerate(commands):
        result = subprocess.run(command, capture_output=True, timeout=60)
        (directory / f'command-{index}.log').write_bytes(
            result.stdout + result.stderr)
        result.check_returncode()
    raw = (directory / 'output.nat').read_bytes()
    sections = list(native.sections(raw))
    fragment_objects = {offset for offset, segment, name, data in sections
                        if name == '__fragment' and segment.startswith('__GPU')}
    fragments = [data for offset, segment, name, data in sections
                 if offset in fragment_objects and segment == '__TEXT'
                 and name == '__text']
    receipt['sections'] = native.extract_sections(directory, raw)
    receipt['native_sha256'] = native.sha(raw)
    receipt['source_sha256'] = native.sha(source.encode())
    receipt['pack_matches'] = [
        i for code in fragments for i in range(len(code))
        if code[i:i + len(PACK)] == PACK]
    (directory / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    native.require(len(fragments) == 1 and len(receipt['pack_matches']) == 1,
                   'Native sample staging changed; inspect captured evidence')
    print(f'{name}: native register-8 PACK_SMPIR verified')


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    capture(output, 'sample-fragment-x', '(uint(position.x) & 3u)')
    capture(output, 'sample-fragment-xy',
            '((uint(position.x) + uint(position.y)) & 3u)')


if __name__ == '__main__':
    main()

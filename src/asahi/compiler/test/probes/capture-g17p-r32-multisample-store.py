#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Capture Apple's single-target 32-bit tile store for a multisample read.

agx3_emit_tile() refuses a store whose value sits at half 16 or above unless
the store is part of a quad or multiple-render-target group. A four-sample R32
background reload is exactly the unwitnessed shape: its value cannot sit lower,
since the texture staging banks hold the low halves, so
KHR-GL33.texture_size_promotion cannot render to R32F or R32UI.

This asks Apple's own compiler for the same shape -- one colour attachment, a
multisample read, a single 32-bit channel out -- at one sample and at four, and
keeps its native code so the tile store can be read rather than inferred.

What it has established so far:

  r32uint, one sample   e70654000000012200000000
  r32float, one sample  e71654000000012200000000
  r32uint, four samples e71654000000022208000008
  r32float, four samples (identical to the uint one)

agx3_tile_store() reproduces the one-sample uint word exactly, at
data=half0 width=W32 mask=1 offset=0 sample=immediate1, so the scalar 32-bit
store itself is already grounded. The four-sample word differs from what this
encoder emits for the same parameters with a register sample mask in exactly
two bits:

  bit 12 (byte 1 bit 4)  -- also the bit that separates Apple's one-sample
                            float store from its one-sample uint store, yet
                            set in both four-sample stores
  bit 67 (byte 8 bit 3)  -- never set by this encoder

Field 13 is not either of them: forcing it lands in byte 7. Naming the two
fields against the lstcol table is what remains before the store can be
grounded for a single target at a high register.

Usage: python3 capture-g17p-r32-multisample-store.py NEW_OUTPUT_DIRECTORY
Requires Apple's metal, metallib and metal-nt tools through xcrun.
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
vertex float4 vertex_main(uint id [[vertex_id]]) {
    return float4(float2((id << 1) & 2, id & 2) * 2.0 - 1.0, 0.0, 1.0);
}
'''


def capture(output, name, pixel_format, source_type, swizzle, samples=1):
    directory = output / name
    directory.mkdir()
    source = VERTEX + '''
fragment float fragment_main(float4 position [[position]],
                             ''' + source_type + ''' source [[texture(0)]]) {
    return source.read(uint2(position.xy), uint(position.x) & 3u)''' + \
        swizzle + ''';
}
'''
    (directory / 'input.metal').write_text(source)
    render = {
        'vertex_function': 'vertex_main',
        'fragment_function': 'fragment_main',
        'color_attachments': [{'pixel_format': pixel_format}],
    }
    if samples > 1:
        render['raster_sample_count'] = samples
    pipeline = {'pipelines': {'render_pipelines': [render]}}
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
    fragment_objects = {offset for offset, segment, name_, data in sections
                        if name_ == '__fragment' and segment.startswith('__GPU')}
    fragments = [data for offset, segment, name_, data in sections
                 if offset in fragment_objects and segment == '__TEXT'
                 and name_ == '__text']
    receipt['sections'] = native.extract_sections(directory, raw)
    receipt['native_sha256'] = native.sha(raw)
    receipt['source_sha256'] = native.sha(source.encode())
    receipt['fragment_text'] = [code.hex() for code in fragments]
    (directory / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    native.require(len(fragments) == 1,
                   'expected exactly one fragment object')
    print(f'{name}: {len(fragments[0])} bytes of native fragment text')
    return fragments[0]


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    capture(output, 'r32float-1x', 'R32Float', 'texture2d_ms<float>', '.x')
    capture(output, 'r32uint-1x', 'R32Uint', 'texture2d_ms<uint>', '.x')
    # The shape the background reload reaches: a four-sample colour target,
    # so the store carries a register sample mask.
    capture(output, 'r32float-4x', 'R32Float', 'texture2d_ms<float>', '.x', 4)
    capture(output, 'r32uint-4x', 'R32Uint', 'texture2d_ms<uint>', '.x', 4)


if __name__ == '__main__':
    main()

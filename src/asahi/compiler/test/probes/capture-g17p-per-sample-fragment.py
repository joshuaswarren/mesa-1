#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Capture how G17P runs a fragment function once per sample.

Usage: python3 capture-g17p-per-sample-fragment.py NEW_OUTPUT_DIRECTORY
Requires Apple's metal, metallib and metal-nt tools through xcrun.

Apple does not loop. A fragment function that reads [[sample_id]] at four
rasterization samples is emitted as one program with the body repeated once
per sample -- four staged operand groups and four tilebuffer stores, whose
sample operands differ -- and the ordinary 0x0e terminator. At one sample the
per-sample and per-pixel programs are the same length, which is the collapse
agx_nir_lower_g17p_1x_sample_shading() performs.

This is the witness that AGX3 has no equivalent of the AGX2 external sample
loop spliced by agx_fast_link(): the way to run per-sample on G17P is to
unroll the body, not to wrap it.
"""
import importlib.util, json, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('native_sections', HERE/'capture-g17p-mrt-eot.py')
native = importlib.util.module_from_spec(spec); spec.loader.exec_module(native)
OUT = Path(sys.argv[1]).resolve(); OUT.mkdir(parents=True, exist_ok=True)

VERTEX = '''
#include <metal_stdlib>
using namespace metal;
vertex float4 vertex_main(uint id [[vertex_id]]) {
    return float4(float2((id << 1) & 2, id & 2) * 2.0 - 1.0, 0.0, 1.0);
}
'''

def build(name, fragment, samples):
    d = OUT/name; d.mkdir(exist_ok=True)
    (d/'input.metal').write_text(VERTEX + fragment)
    pipeline = {'pipelines': {'render_pipelines': [{
        'vertex_function': 'vertex_main',
        'fragment_function': 'fragment_main',
        'color_attachments': [{'pixel_format': 'RGBA8Unorm'}],
        'raster_sample_count': samples,
    }]}}
    (d/'pipeline.mtlp-json').write_text(json.dumps(pipeline, indent=2))
    cmds = [
      ['xcrun','metal','-c',str(d/'input.metal'),'-o',str(d/'input.air')],
      ['xcrun','metallib',str(d/'input.air'),'-o',str(d/'input.metallib')],
      ['xcrun','metal-nt','-arch','applegpu_g17p','-platform_version','macos','27.0','27.0',
       '-N',str(d/'pipeline.mtlp-json'),str(d/'input.metallib'),'-o',str(d/'output.nat')],
    ]
    for i,c in enumerate(cmds):
        r = subprocess.run(c, capture_output=True, timeout=120)
        (d/f'command-{i}.log').write_bytes(r.stdout+r.stderr)
        if r.returncode: print(name,'FAILED',c[0],(r.stdout+r.stderr).decode()[:400]); return None
    raw = (d/'output.nat').read_bytes()
    sections = list(native.sections(raw))
    frag_off = {o for o,s,n,dd in sections if n=='__fragment' and s.startswith('__GPU')}
    frags = [dd for o,s,n,dd in sections if o in frag_off and s=='__TEXT' and n=='__text']
    return frags

PER_PIXEL = '''
fragment float4 fragment_main(float4 position [[position]]) {
    return float4(0.25, 0.5, 0.75, 1.0);
}
'''
PER_SAMPLE = '''
fragment float4 fragment_main(float4 position [[position]], uint s [[sample_id]]) {
    return float4((float)s * 0.25, 0.5, 0.75, 1.0);
}
'''
for samples in (1, 4):
    for label, src in (('per-pixel', PER_PIXEL), ('per-sample', PER_SAMPLE)):
        name = f'{label}-{samples}x'
        frags = build(name, src, samples)
        if frags is None: continue
        for i, code in enumerate(frags):
            print(f'{name}: fragment[{i}] {len(code)} bytes  {code.hex()}')

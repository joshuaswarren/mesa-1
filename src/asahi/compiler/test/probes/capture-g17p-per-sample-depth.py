#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Capture a G17P fragment that writes depth, per pixel and per sample.

Usage: python3 capture-g17p-per-sample-depth.py NEW_OUTPUT_DIRECTORY
Requires Apple's metal, metallib and metal-nt tools through xcrun.

The companion to capture-g17p-per-sample-fragment.py, for the depth store
rather than the tilebuffer one. See README-g17p-per-sample-fragment.md.
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
struct DepthOutput { float depth [[depth(any)]]; };
vertex float4 vertex_main(uint id [[vertex_id]]) {
    return float4(float2((id << 1) & 2, id & 2) * 2.0 - 1.0, 0.0, 1.0);
}
'''
def build(name, fragment, samples, colour):
    d = OUT/name; d.mkdir(exist_ok=True)
    (d/'input.metal').write_text(VERTEX + fragment)
    rp = {'vertex_function':'vertex_main','fragment_function':'fragment_main',
          'depth_attachment_pixel_format':'Depth32Float','raster_sample_count':samples}
    if colour: rp['color_attachments'] = [{'pixel_format':'RGBA8Unorm'}]
    (d/'pipeline.mtlp-json').write_text(json.dumps({'pipelines':{'render_pipelines':[rp]}}, indent=2))
    cmds = [['xcrun','metal','-c',str(d/'input.metal'),'-o',str(d/'input.air')],
            ['xcrun','metallib',str(d/'input.air'),'-o',str(d/'input.metallib')],
            ['xcrun','metal-nt','-arch','applegpu_g17p','-platform_version','macos','27.0','27.0',
             '-N',str(d/'pipeline.mtlp-json'),str(d/'input.metallib'),'-o',str(d/'output.nat')]]
    for i,c in enumerate(cmds):
        r = subprocess.run(c, capture_output=True, timeout=120)
        (d/f'command-{i}.log').write_bytes(r.stdout+r.stderr)
        if r.returncode: print(name,'FAILED:',(r.stdout+r.stderr).decode()[:300]); return
    raw = (d/'output.nat').read_bytes(); sec = list(native.sections(raw))
    off = {o for o,s,n,dd in sec if n=='__fragment' and s.startswith('__GPU')}
    for dd in [dd for o,s,n,dd in sec if o in off and s=='__TEXT' and n=='__text']:
        print(f'{name}: {len(dd)} bytes  {dd.hex()}')

DEPTH_PIXEL = '''
fragment DepthOutput fragment_main(float4 position [[position]],
                                   depth2d_ms<float> src [[texture(0)]]) {
    return {src.read(uint2(position.xy), 0)};
}
'''
DEPTH_SAMPLE = '''
fragment DepthOutput fragment_main(float4 position [[position]],
                                   uint s [[sample_id]],
                                   depth2d_ms<float> src [[texture(0)]]) {
    return {src.read(uint2(position.xy), s)};
}
'''
for samples in (1, 2, 4):
    build(f'depth-pixel-{samples}x', DEPTH_PIXEL, samples, False)
    build(f'depth-sample-{samples}x', DEPTH_SAMPLE, samples, False)

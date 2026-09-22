#!/usr/bin/env python3
"""Compile exact CTS mask/alpha expressions for G17P; never execute the GPU."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=False)
source = '''#include <metal_stdlib>
using namespace metal;
vertex float4 vertex_main(uint id [[vertex_id]]) {
  return float4(float2((id << 1) & 2, id & 2) * 2.0 - 1.0, 0.5, 1.0);
}
struct Output { float4 colour [[color(0)]]; uint mask [[sample_mask]]; };
fragment Output fragment_main(float4 pos [[position]]) {
  return {float4(0.0, 0.0, 1.0, int(pos.x) < 2 ? 0.0 : 1.0),
          int(pos.y) >= 2 ? 0u : 255u};
}
'''
(out / 'input.metal').write_text(source)
for mode in ('sample-mask', 'alpha-coverage', 'conditional-kill'):
    atc = mode == 'alpha-coverage'
    d = out / mode
    d.mkdir()
    case_source = source
    if mode == 'conditional-kill':
        case_source = source[:source.index('struct Output')] + '''
fragment float4 fragment_main(float4 pos [[position]]) {
  if (int(pos.y) >= 2) discard_fragment();
  return float4(0.0, 0.0, 1.0, int(pos.x) < 2 ? 0.0 : 1.0);
}
'''
    (d / 'input.metal').write_text(case_source)
    pipeline = {'pipelines': {'render_pipelines': [{
        'vertex_function': 'vertex_main', 'fragment_function': 'fragment_main',
        'color_attachments': [{'pixel_format': 'RGBA8Unorm'}],
        'depth_attachment_pixel_format': 'Depth32Float',
        'raster_sample_count': 4, 'alpha_to_coverage_enabled': atc,
    }]}}
    (d / 'pipeline.mtlp-json').write_text(json.dumps(pipeline, indent=2))
    commands = [
        ['xcrun', 'metal', '-c', str(d/'input.metal'), '-o', str(d/'input.air')],
        ['xcrun', 'metallib', str(d/'input.air'), '-o', str(d/'input.metallib')],
        ['xcrun', 'metal-nt', '-arch', 'applegpu_g17p', '-platform_version',
         'macos', '27.0', '27.0', '-N', str(d/'pipeline.mtlp-json'),
         str(d/'input.metallib'), '-o', str(d/'output.nat')],
    ]
    for i, command in enumerate(commands):
        p = subprocess.run(command, capture_output=True, timeout=60)
        (d / f'command-{i}.log').write_bytes(p.stdout + p.stderr)
        if p.returncode:
            raise RuntimeError((p.stdout+p.stderr).decode())
    print(d.name, hashlib.sha256((d/'output.nat').read_bytes()).hexdigest())

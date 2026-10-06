#!/usr/bin/env python3
# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT
"""Compare a D3D12HelloWorld frame dumped by aurora_sample_main --dump against
a CPU reference of the sample's own geometry.

usage: sample-reference.py <Sample> <frame.ppm> [frames]

<Sample> is HelloTriangle, HelloBundles, HelloConstBuffers or HelloTexture.
[frames] is the --frames count of the run; HelloConstBuffers moves its
triangle by 0.005 clip units per OnUpdate, so the dumped frame depends on it.

The reference follows the D3D11+ rasterisation rules the samples rely on:
pixel centres at +0.5, top-left fill, the viewport covering the whole back
buffer, linear (w = 1) attribute interpolation, UNORM rounding to nearest.
HelloTexture samples its 256x256 checkerboard with a point filter and a
transparent-black border. A pixel matches when every channel is within
--tolerance (default 2) of the reference. Coverage that differs only on a
pixel whose centre lies within 1e-3 px of an edge, and checkerboard colours
within 1e-3 texels of a cell boundary, are counted separately as ambiguous
rather than wrong. Prints one JSON line; exit status 0 when nothing is wrong.
"""
import argparse
import json
import re
import struct
import sys

CLEAR = (0.0, 0.2, 0.4)


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def unorm8(v):
    return int(min(max(v, 0.0), 1.0) * 255.0 + 0.5)


def read_ppm(path):
    with open(path, 'rb') as f:
        data = f.read()
    # Consume exactly the header separator, preserving even a whitespace
    # first pixel byte. The sample harness emits this comment-free P6 form.
    header = re.match(rb'P6\s+(\d+)\s+(\d+)\s+255(?:\r\n|[ \t\r\n])', data)
    if not header:
        raise ValueError('not a sample-harness 8-bit P6 PPM')
    w, h = int(header[1]), int(header[2])
    pixels = data[header.end():]
    if w <= 0 or h <= 0 or len(pixels) != w * h * 3:
        raise ValueError('invalid dimensions or PPM pixel byte count')
    return w, h, pixels


def const_buffer_offset(frames):
    x = f32(0.0)
    for _ in range(frames):
        x = f32(x + f32(0.005))
        if x > 1.25:
            x = f32(-1.25)
    return x


def geometry(sample, w, h, frames):
    ar = f32(w / h)
    top, right, left = (0.0, f32(0.25 * ar)), (0.25, f32(-0.25 * ar)), (-0.25, f32(-0.25 * ar))
    if sample == 'HelloTexture':
        attrs = [(0.5, 0.0), (1.0, 1.0), (0.0, 1.0)]
    else:
        attrs = [(1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)]
    pos = [top, right, left]
    if sample == 'HelloConstBuffers':
        dx = const_buffer_offset(frames)
        pos = [(x + dx, y) for x, y in pos]
    # Viewport transform to pixels.
    px = [((x * 0.5 + 0.5) * w, (0.5 - y * 0.5) * h) for x, y in pos]
    return px, attrs


def checker(u, v):
    """Point-sampled HelloTexture texel, or None when within 1e-3 texel of a
    cell edge (ambiguous)."""
    if not (0.0 <= u < 1.0 and 0.0 <= v < 1.0):
        return (0.0, 0.0, 0.0)  # transparent-black border, blended off
    tx, ty = u * 256.0, v * 256.0
    for t in (tx, ty):
        frac = (t / 32.0) % 1.0
        if min(frac, 1.0 - frac) * 32.0 < 1e-3:
            return None
    i, j = int(tx) // 32, int(ty) // 32
    return (0.0, 0.0, 0.0) if i % 2 == j % 2 else (1.0, 1.0, 1.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sample', type=lambda s: s.removeprefix('D3D12'),
                        choices=['HelloTriangle', 'HelloBundles',
                                 'HelloConstBuffers', 'HelloTexture'])
    parser.add_argument('path')
    parser.add_argument('frames', type=int, nargs='?', default=600)
    parser.add_argument('--tolerance', type=int, default=2)
    args = parser.parse_args()
    if args.frames <= 0 or not 0 <= args.tolerance <= 255:
        parser.error('frames must be positive and tolerance must be in [0, 255]')
    sample, frames, tol = args.sample, args.frames, args.tolerance
    try:
        w, h, pix = read_ppm(args.path)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    p, attrs = geometry(sample, w, h, frames)

    def edge(a, b, x, y):
        return (p[b][0] - p[a][0]) * (y - p[a][1]) - (p[b][1] - p[a][1]) * (x - p[a][0])

    area = edge(0, 1, p[2][0], p[2][1])
    clear = tuple(unorm8(c) for c in CLEAR)
    stats = dict(pixels=w * h, covered_ref=0, wrong=0, coverage_wrong=0, ambiguous=0)
    first_wrong = []
    xs = [q[0] for q in p]
    ys = [q[1] for q in p]
    x0, x1 = max(int(min(xs)) - 1, 0), min(int(max(xs)) + 2, w)
    y0, y1 = max(int(min(ys)) - 1, 0), min(int(max(ys)) + 2, h)
    for y in range(h):
        for x in range(w):
            o = (y * w + x) * 3
            got = (pix[o], pix[o + 1], pix[o + 2])
            want = clear
            near_edge = False
            if x0 <= x < x1 and y0 <= y < y1:
                cx, cy = x + 0.5, y + 0.5
                e = [edge(1, 2, cx, cy), edge(2, 0, cx, cy), edge(0, 1, cx, cy)]
                if area < 0:
                    e = [-v for v in e]
                lens = []
                for k, (a, b) in enumerate(((1, 2), (2, 0), (0, 1))):
                    dx, dy = p[b][0] - p[a][0], p[b][1] - p[a][1]
                    lens.append((dx * dx + dy * dy) ** 0.5)
                near_edge = any(abs(e[k]) / lens[k] < 1e-3 for k in range(3))
                inside = all(v > 0 for v in e)
                if inside or (near_edge and all(v >= 0 for v in e)):
                    stats['covered_ref'] += inside
                    bary = [v / abs(area) for v in e]
                    a = [sum(bary[i] * attrs[i][c] for i in range(3))
                         for c in range(len(attrs[0]))]
                    if sample == 'HelloTexture':
                        t = checker(a[0], a[1])
                        if t is None:
                            stats['ambiguous'] += 1
                            continue
                        want = tuple(unorm8(c) for c in t)
                    else:
                        want = tuple(unorm8(c) for c in a[:3])
            if any(abs(got[c] - want[c]) > tol for c in range(3)):
                if near_edge:
                    stats['ambiguous'] += 1
                    continue
                stats['wrong'] += 1
                if (got == clear) != (want == clear):
                    stats['coverage_wrong'] += 1
                if len(first_wrong) < 5:
                    first_wrong.append(dict(x=x, y=y, got=got, want=want))
    stats.update(sample=sample, width=w, height=h, frames=frames, tolerance=tol,
                 first_wrong=first_wrong,
                 verdict='MATCH' if stats['wrong'] == 0 and stats['covered_ref'] else 'MISMATCH')
    print(json.dumps(stats))
    return 0 if stats['verdict'] == 'MATCH' else 1


if __name__ == '__main__':
    sys.exit(main())

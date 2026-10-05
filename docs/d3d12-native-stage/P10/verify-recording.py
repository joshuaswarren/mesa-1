#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check retained P10 uploads and CDM records independently of the recorder."""
import argparse
import json
from pathlib import Path
import struct


def check(directory, name):
    meta = json.loads((directory / f'P10-{name}.json').read_text())
    assert meta['registered_recording'] and not meta['physical_admission']
    assert meta['render_placeholder'] and not meta['gpu_submitted']
    assert meta['draws'] == 2
    indirect = name == 'indirect'
    assert meta['indirect_local'] == indirect
    assert meta['render_header'] == {'vdm_barrier': 0, 'cdm_barrier': 1}
    uploads = (directory / f'P10-{name}.uploads').read_bytes()
    origin = meta['stream_base'] + meta['upload_offset']
    ioto = (directory / f'P10-{name}.Ioto').read_bytes()

    def span(address, size):
        offset = address - origin
        assert 0 <= offset <= len(uploads) - size
        return uploads[offset:offset + size]

    def pointer(cell):
        assert cell % 8 == 0
        return struct.unpack('<Q', span(cell, 8))[0]

    vp = [pointer(cell) for cell in meta['vp_cells']]
    gp = [pointer(cell) for cell in meta['gp_cells']]
    assert vp[0] != vp[1] and gp[0] != gp[1]
    grids = []
    for index in range(2):
        vertex = span(vp[index], 64)
        geometry = span(gp[index], 316)
        # Packed Poly ABI: vp.grid at byte20, gp.grid at byte256.
        expected = (0, 0, 1, 64, 1, 1) if indirect else (4, index + 1, 1, 64, 1, 1)
        assert struct.unpack_from('<6I', vertex, 20) == expected
        assert struct.unpack_from('<6I', geometry, 256) == expected
        output = struct.unpack_from('<Q', vertex, 48)[0]
        if indirect:
            assert output == 0  # setup has not executed; do not invent GPU counts
        else:
            assert struct.unpack_from('<I', vertex, 16)[0] == 4
            assert struct.unpack_from('<2I', geometry, 280) == (4, index + 1)
            assert struct.unpack_from('<Q', vertex, 56)[0] == 0x100000001
            span(output, 4 * (index + 1) * 2 * 16)
        grids += [vp[index] + 20, gp[index] + 256]

    stream = (directory / f'P10-{name}.CDM').read_bytes()
    sizes = [40, 40] * 2 if not indirect else [40, 24, 24] * 2
    assert len(stream) == sum(size + 4 for size in sizes) + 4
    offset = 0
    indirect_grids = []
    programs = []
    direct_grids = []
    for size in sizes:
        words = struct.unpack_from('<' + 'I' * (size // 4), stream, offset)
        assert bool(words[0] & 0x10000000) == (size == 24)
        address = (words[2] << 6) | ((words[3] & 0xff) << 40)
        assert (words[3] & 0xffffff00) == 0x40000000
        assert address % 64 == 0 and 0 <= address - meta['usc_base'] < len(ioto)
        programs.append(address)
        if size == 24:
            indirect_grids.append((words[4] << 32) | words[5])
        else:
            direct_grids.append(words[4:10])
        # Shared strong flags=5 barrier and canonical four-byte terminator.
        assert struct.unpack_from('<I', stream, offset + size)[0] == 0x60001560
        offset += size + 4
    assert struct.unpack_from('<I', stream, offset)[0] == 0x40000000
    assert len(set(programs)) == meta['dispatches'] == len(sizes)
    if indirect:
        assert indirect_grids == grids
        assert direct_grids == [(1, 1, 1, 1, 1, 1)] * 2
    else:
        assert direct_grids == [(4, 1, 1, 64, 1, 1)] * 2 + [(4, 2, 1, 64, 1, 1)] * 2
    return dict(name=name, dispatches=len(sizes), strong_barriers=len(sizes),
                program_addresses=programs, indirect_grid_addresses=indirect_grids,
                gpu_submitted=False, physical_admission=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps([check(args.directory, name) for name in ('direct', 'indirect')], indent=2))

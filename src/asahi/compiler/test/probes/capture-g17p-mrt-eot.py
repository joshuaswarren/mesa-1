#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Reproduce and validate internal G17P EOT compiler witnesses; no GPU use.

capture DIR: invoke metal-nt into a new directory, retaining inputs and logs.
check DIR: verify native input/text/metadata and the current Mesa source.
header DIR: emit the RGBA8 C header after the same native validation.

This is an offline developer tool, not part of a Mesa build or runtime.
The JSON recipe pins semantic inputs and executable text, not object packaging.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()

def sections(data):
    """Enumerate bounded Mach-O sections, including nested GPU objects."""
    start = 0
    while True:
        start = data.find(b'\xcf\xfa\xed\xfe', start)
        if start < 0:
            return
        if start + 32 > len(data):
            return
        ncmds, sizecmds = struct.unpack_from('<II', data, start + 16)
        pos = start + 32
        end = pos + sizecmds
        if ncmds > 4096 or end > len(data):
            start += 4
            continue
        for _ in range(ncmds):
            if pos + 8 > end:
                break
            cmd, size = struct.unpack_from('<II', data, pos)
            if size < 8 or pos + size > end:
                break
            if cmd == 0x19 and size >= 72:
                count = struct.unpack_from('<I', data, pos + 64)[0]
                for j in range(min(count, (size - 72) // 80)):
                    entry = pos + 72 + 80 * j
                    name = data[entry:entry+16].split(b'\0')[0].decode('ascii')
                    segment = data[entry+16:entry+32].split(b'\0')[0].decode('ascii')
                    length, offset = struct.unpack_from('<QI', data, entry + 40)
                    begin = start + offset
                    if length and begin + length <= len(data):
                        yield start, segment, name, data[begin:begin+length]
            pos += size
        start += 4

def extract_sections(directory, raw):
    result = []
    for i, (offset, segment, name, contents) in enumerate(sections(raw)):
        if name != '__text' and not segment.startswith('__GPU'):
            continue
        filename = f'{offset:06x}-{i}-{segment}-{name}.bin'
        (directory/filename).write_bytes(contents)
        result.append({'object_offset': offset, 'segment': segment, 'name': name, 'file': filename,
                       'size': len(contents), 'sha256': sha(contents)})
    return result


def field(data, table, index):
    vtable = table - struct.unpack_from('<i', data, table)[0]
    require(0 <= vtable <= len(data) - 4, 'metadata vtable bounds')
    size = struct.unpack_from('<H', data, vtable)[0]
    require(size >= 4 and not size % 2 and vtable + size <= len(data),
            'metadata vtable length')
    if 4 + 2 * index >= size:
        return None
    offset = struct.unpack_from('<H', data, vtable + 4 + 2 * index)[0]
    require(not offset or 0 <= table + offset < len(data), 'metadata field bounds')
    return table + offset if offset else None


def u32(data, offset):
    return struct.unpack_from('<I', data, offset)[0]


def scalar(data, table, index, byte=False):
    offset = field(data, table, index)
    return (data[offset] if byte else u32(data, offset)) if offset else 0


def pointer(data, table, index):
    offset = field(data, table, index)
    require(offset is not None, 'missing metadata pointer')
    result = offset + u32(data, offset)
    require(result <= len(data) - 4, 'metadata pointer bounds')
    return result


def tables(data, table, index):
    vector = pointer(data, table, index)
    count = u32(data, vector)
    require(vector + 4 + 4 * count <= len(data), 'metadata vector bounds')
    return [entry + u32(data, entry)
            for entry in range(vector + 4, vector + 4 + 4 * count, 4)]


def builtin_input(recipe, variant):
    seed = bytes.fromhex((HERE / 'g17p-native-rt0-eot.btnf.hex').read_text())
    require(len(seed) == 123 and sha(seed) == recipe['seed_sha256'], 'seed mismatch')
    count = len(variant['formats'])
    require(1 <= count <= 8 and len(variant['components']) == count,
            'attachment count')
    require(recipe['physical_partition'] == 0, 'compact EOT partition must be zero')
    # Preserve the recorded BTNF root, union and attachment schema. Offset80
    # is the active mask; the attachment vector begins at84. Each attachment
    # table has byte fields: partition(+4), format(+5), dimension(+6),
    # flags(+8), components(+9). Format3 is U8NORM; format8 is F16.
    data = bytearray(seed[:84])
    data[80] = (1 << count) - 1
    data += struct.pack('<I', count) + bytes(4 * count)
    for index, (fmt, comps) in enumerate(zip(variant['formats'], variant['components'])):
        data += seed[92:112]
        table = len(data)
        data += seed[112:123]
        data[table + 4] = 0
        data[table + 5] = fmt
        data[table + 9] = comps
        entry = 88 + 4 * index
        struct.pack_into('<I', data, entry, table - entry)
        data += bytes(-len(data) % 4)
    require(sha(data) == variant['input_sha256'], 'serialized input differs from witness')
    return bytes(data)


def read_capture(directory, recipe, variant):
    require((directory / 'input.btnf').read_bytes() == builtin_input(recipe, variant),
            'captured input mismatch')
    receipt = json.loads((directory / 'receipt.json').read_text())
    require(receipt['exit_code'] == 0 and receipt['hardware_execution'] is False,
            'capture did not complete offline')
    raw = (directory / 'output.nat').read_bytes()
    require(sha(raw) == receipt['native_sha256'], 'native object digest mismatch')
    extracted = list(sections(raw))
    def one(segment, name):
        found = [(base, contents) for base, seg, key, contents in extracted
                 if seg == segment and key == name]
        require(len(found) == 1, 'expected exactly one native ' + segment + ',' + name)
        return found[0]
    base, code = one('__TEXT', '__text')
    meta_base, data = one('__GPU_METADATA', '__fragment')
    load_base, load = one('__GPU_LD_MD', '__fragment')
    require(base == meta_base == load_base, 'native sections belong to different objects')
    require(len(code) == variant['text_bytes'] and sha(code) == variant['text_sha256'],
            'native executable differs from pinned witness')
    count = len(variant['formats'])
    common = pointer(data, u32(data, 0), 0)
    require(scalar(data, common, 0) == variant['native_gpr_dwords'], 'native GPR metadata')
    require(scalar(data, common, 38) == scalar(data, common, 42) == variant['staging_bytes'],
            'native staging metadata')
    esl = tables(data, common, 2)
    require(len(esl) == 2, 'native ESL count')
    require(tuple(scalar(data, esl[0], i, byte=(i == 0)) for i in (0,2,3,4)) == (3,2,0,0),
            'native ESL pointer contract')
    require(tuple(scalar(data, esl[1], i, byte=(i == 0)) for i in (0,2,3,4)) == (0,2*count-1,2,0),
            'native ESL offset contract')
    args = tables(data, common, 10)
    require(len(args) == count, 'native offset argument count')
    for i, arg in enumerate(args):
        require((scalar(data, arg, 0, byte=True), scalar(data, arg, 1), scalar(data, arg, 2))
                == (5,i,2*i), 'native compact attachment argument order')
    load_common = pointer(load, u32(load, 0), 0)
    require(scalar(load, load_common, 11) == (1 << count) - 1, 'native RT store mask')
    return code


def capture(directory, recipe, variant):
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'input.btnf').write_bytes(builtin_input(recipe, variant))
    command = ['xcrun', 'metal-nt', '-arch', recipe['architecture'],
               '-platform_version', *recipe['platform'],
               '-N', str(directory / 'input.btnf'), '-o', str(directory / 'output.nat')]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    (directory / 'compiler.log').write_bytes(result.stdout)
    receipt = {'command': command, 'hardware_execution': False, 'exit_code': result.returncode}
    if result.returncode == 0:
        raw = (directory / 'output.nat').read_bytes()
        receipt.update(native_sha256=sha(raw), sections=extract_sections(directory, raw))
    (directory / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    read_capture(directory, recipe, variant)


def render_header(recipe, codes):
    parts = ['/* SPDX-License-Identifier: MIT */',
             '/* Native G17P EOT texts and metadata captured with MetalToolchain',
             ' * v27.1.5194.15. See compiler/test/probes/README-g17p-wide-mrt.md.',
             ' * Every attachment stays in physical imageblock partition zero.',
             ' * EOT partitions are offset/sample-stride quotients, distinct from',
             ' * fragment pixel synchronization classes. Active PBEs are packed. */',
             '#ifndef AGX_G17P_WIDE_MRT_EOT_H', '#define AGX_G17P_WIDE_MRT_EOT_H',
             '#include <stdbool.h>', '#include <stdint.h>', '']
    entries = []

    for count in range(1, 9):
        variant = next(v for v in recipe['variants'] if v['name'] == f'rgba8-{count}')
        code = codes[variant['name']]
        parts += [f'/* text SHA256 {variant["text_sha256"]} */',
                  f'static const uint8_t agx_g17p_wide_mrt_eot_{count}[] = {{']
        parts += ['   '+','.join(f'0x{b:02x}' for b in code[i:i+12])+',' for i in range(0,len(code),12)]
        parts += ['};', '']
        entries.append((count, variant['staging_bytes'] // 2))

    parts += ['struct agx_g17p_wide_mrt_eot {', '   const uint8_t *code;',
              '   uint16_t size;', '   uint16_t staging_halfs;', '};', '',
              'static inline bool', 'agx_g17p_wide_mrt_eot_get(unsigned count, struct agx_g17p_wide_mrt_eot *out)',
              '{', '   if (!out)', '      return false;', '   switch (count) {']
    for count, staging in entries:
        parts += [f'   case {count}:',
                  f'      *out = (struct agx_g17p_wide_mrt_eot){{agx_g17p_wide_mrt_eot_{count},',
                  f'         sizeof(agx_g17p_wide_mrt_eot_{count}), {staging}}};',
                  '      return true;']
    parts += ['   default:', '      return false;', '   }', '}', '#endif']
    return '\n'.join(parts) + '\n'


def embedded_array(path, name):
    match = re.search(re.escape(name) + r'\[\] = \{(.*?)\};', path.read_text(), re.S)
    require(match is not None, 'missing source array ' + name)
    return bytes(int(value, 16) for value in re.findall(r'0x([0-9a-fA-F]{2})', match[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['capture', 'check', 'header'])
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    recipe = json.loads((HERE / 'g17p-mrt-eot.json').read_text())
    directory = args.directory.resolve()
    if args.mode == 'capture':
        directory.mkdir(parents=True, exist_ok=False)
        version = subprocess.run(['xcrun', 'metal', '--version'], capture_output=True, text=True, check=True)
        (directory / 'toolchain.txt').write_text(version.stdout)
        for variant in recipe['variants']:
            capture(directory / variant['name'], recipe, variant)
            print(variant['name'] + ': captured and validated', flush=True)
    codes = {v['name']: read_capture(directory / v['name'], recipe, v) for v in recipe['variants']}
    header = render_header(recipe, codes)
    if args.mode == 'header':
        print(header, end='')
    elif args.mode == 'check':
        require(header == (REPO / 'src/asahi/lib/agx_g17p_wide_mrt_eot.h').read_text(),
                'generated RGBA8 header differs from source')
        require(codes['rgba16f-r8'] == embedded_array(
            REPO / 'src/asahi/compiler/agx3_pack_binary.c', 'g17p_wboit_mrt_eot'),
            'mixed-format production EOT differs from native witness')
        require(codes['rgba16f-r8'] == embedded_array(
            REPO / 'src/asahi/compiler/test/test-agx3-graphics-compile.c', 'expected_mrt'),
            'mixed-format test oracle differs from native witness')
        print('Validated 9 native inputs, executable texts, resource metadata and source arrays.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, StopIteration, struct.error, subprocess.SubprocessError) as error:
        sys.exit('EOT reproduction failed: ' + str(error))

#!/usr/bin/env python3
"""Generate native G17P builtin EOT texts for chosen attachment dimensions.
Reuses the recorded BTNF schema from capture-g17p-mrt-eot.py; no GPU use."""
import hashlib, json, os, struct, subprocess, sys, tempfile
from pathlib import Path
P = Path(__file__).resolve().parent
OUT = Path(os.environ.get('AGX_PROBE_OUT') or
           tempfile.mkdtemp(prefix='g17p-array-eot-'))
def sha(d): return hashlib.sha256(d).hexdigest()
def u32(d,o): return struct.unpack_from('<I',d,o)[0]

def sections(data):
    start = 0
    while True:
        start = data.find(b'\xcf\xfa\xed\xfe', start)
        if start < 0: return
        if start + 32 > len(data): return
        ncmds, sizecmds = struct.unpack_from('<II', data, start + 16)
        pos = start + 32; end = pos + sizecmds
        if ncmds > 4096 or end > len(data):
            start += 4; continue
        for _ in range(ncmds):
            if pos + 8 > end: break
            cmd, size = struct.unpack_from('<II', data, pos)
            if size < 8 or pos + size > end: break
            if cmd == 0x19 and size >= 72:
                count = u32(data, pos + 64)
                for j in range(min(count, (size - 72)//80)):
                    e = pos + 72 + 80*j
                    name = data[e:e+16].split(b'\0')[0].decode()
                    seg  = data[e+16:e+32].split(b'\0')[0].decode()
                    length, offset = struct.unpack_from('<QI', data, e + 40)
                    begin = start + offset
                    if length and begin + length <= len(data):
                        yield seg, name, data[begin:begin+length]
            pos += size
        start += 4

SEED = bytes.fromhex((P/'g17p-native-rt0-eot.btnf.hex').read_text())
assert len(SEED) == 123, len(SEED)

def builtin_input(formats, comps, dim):
    """Same construction as capture-g17p-mrt-eot.py builtin_input(), with the
    attachment dimension byte (+6) parameterized."""
    count = len(formats)
    data = bytearray(SEED[:84])
    data[80] = (1 << count) - 1
    data += struct.pack('<I', count) + bytes(4*count)
    for index,(fmt,c) in enumerate(zip(formats, comps)):
        data += SEED[92:112]
        table = len(data)
        data += SEED[112:123]
        data[table+4] = 0
        data[table+5] = fmt
        data[table+6] = dim
        data[table+9] = c
        entry = 88 + 4*index
        struct.pack_into('<I', data, entry, table - entry)
        data += bytes(-len(data) % 4)
    return bytes(data)

def run(formats, comps, dim, tag):
    d = OUT/tag
    d.mkdir(parents=True, exist_ok=True)
    inp = builtin_input(formats, comps, dim)
    (d/'input.btnf').write_bytes(inp)
    cmd = ['xcrun','metal-nt','-arch','applegpu_g17p','-platform_version','macos','27.0','27.0',
           '-N',str(d/'input.btnf'),'-o',str(d/'output.nat')]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
    (d/'compiler.log').write_bytes(r.stdout)
    if r.returncode != 0:
        return None, r.stdout.decode(errors='replace')[:500]
    raw = (d/'output.nat').read_bytes()
    text = [c for s,n,c in sections(raw) if s=='__TEXT' and n=='__text']
    if len(text)!=1: return None, 'text sections=%d' % len(text)
    (d/'text.hex').write_text(text[0].hex())
    return text[0], sha(inp)

if __name__ == '__main__':
    # Validation: reproduce the recorded non-array and array captures exactly.
    checks = [
      ('g17p-native-raw32-mrt4.btnf.hex', [10]*4, [4]*4, 2, 'chk-raw32-4-d2'),
      ('g17p-native-raw32-mrt4-array.btnf.hex', [10]*4, [4]*4, 3, 'chk-raw32-4-d3'),
      ('g17p-native-rgba8-mrt4-array.btnf.hex', [3]*4, [4]*4, 3, 'chk-rgba8-4-d3'),
      ('g17p-native-raw32-mrt2-array.btnf.hex', [10]*2, [4]*2, 3, 'chk-raw32-2-d3'),
      ('g17p-native-raw32-mrt3-array.btnf.hex', [10]*3, [4]*3, 3, 'chk-raw32-3-d3'),
    ]
    ok = True
    for name, fmts, cs, dim, tag in checks:
        want_in = bytes.fromhex((P/name).read_text().replace('\n',''))
        got_in = builtin_input(fmts, cs, dim)
        m = (want_in == got_in)
        print(f'{name}: input reproduced={m}')
        ok &= m
        tf = name.replace('.btnf.hex','.text.hex')
        if (P/tf).exists():
            text,_ = run(fmts, cs, dim, tag)
            want_text = bytes.fromhex((P/tf).read_text().replace('\n',''))
            m2 = text == want_text
            print(f'   text reproduced={m2} len={len(text) if text else None} want={len(want_text)}')
            ok &= m2
    print('VALIDATION', 'PASS' if ok else 'FAIL')

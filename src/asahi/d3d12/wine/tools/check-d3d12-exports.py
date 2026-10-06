#!/usr/bin/env python3
# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT
"""Assert Windows D3D12 named exports and ordinals in a built PE using LLVM."""

import argparse
import json
import re
import subprocess


EXPORTS = {
    101: 'D3D12CreateDevice',
    102: 'D3D12GetDebugInterface',
    103: 'D3D12CreateRootSignatureDeserializer',
    104: 'D3D12CreateVersionedRootSignatureDeserializer',
    105: 'D3D12EnableExperimentalFeatures',
    106: 'D3D12GetInterface',
    107: 'D3D12SerializeRootSignature',
    108: 'D3D12SerializeVersionedRootSignature',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dll')
    parser.add_argument('--llvm-readobj', default='llvm-readobj')
    args = parser.parse_args()
    dump = subprocess.check_output(
        [args.llvm_readobj, '--coff-exports', args.dll], text=True)
    exports = {}
    for block in re.findall(r'Export \{(.*?)\}', dump, re.DOTALL):
        ordinal = re.search(r'Ordinal: (\d+)', block)
        name = re.search(r'Name: (\S+)', block)
        rva = re.search(r'RVA: (0x[0-9A-Fa-f]+)', block)
        if ordinal and name and rva and int(rva[1], 16):
            exports[int(ordinal[1])] = name[1]
    print(json.dumps({'dll': args.dll, 'exports': exports}, indent=2))
    if exports != EXPORTS:
        parser.exit(1, 'FAIL: expected exactly the eight named exports at 101-108\n')
    print('PASS: Windows D3D12 ordinals 101-108 and named exports')


if __name__ == '__main__':
    main()

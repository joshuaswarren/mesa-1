#!/usr/bin/env python3
# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT
"""Verify version hashes, allowed fixture delta, signatures and (optionally) Git origin."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
spec = importlib.util.spec_from_file_location('signatures', Path(__file__).with_name('dump-signatures.py'))
signatures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(signatures)


def hashes(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob('*')) if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--git-origin', action='store_true')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    manifest = json.loads((here / 'manifest.json').read_text())
    snapshots = {}
    for version in ('v1', 'v2'):
        snapshot = snapshots[version] = hashes(here / version)
        expected = manifest['versions'][version]['inputs']
        assert snapshot == expected, f'{version}: input hash/file set mismatch'
        checksums = {path: sha for sha, path in
                     (line.split('  ', 1) for line in (here / (version + '.sha256')).read_text().splitlines())}
        assert snapshot == checksums, f'{version}: checksum manifest mismatch'
        print(f'{version}: {len(snapshot)} inputs verified')
    changed = sorted(k for k in snapshots['v1'].keys() | snapshots['v2'].keys()
                     if snapshots['v1'].get(k) != snapshots['v2'].get(k))
    allowed = sorted(manifest['changed_inputs'])
    assert changed == allowed, f'unexpected fixture delta: {changed}'
    assert len(manifest['suites']) == manifest['required_suites'] == 17
    assert set(manifest['corrections']) == {'g1_smoke', 'g5_depth', 'g6_geometry',
                                          'g7_tessellation', 'g8_mesh', 'g9_amplification'}
    for suite in manifest['suites']:
        if suite['name'] not in manifest['corrections']:
            assert snapshots['v1'][suite['source']] == snapshots['v2'][suite['source']]
            assert suite['shader_args']['v1'] == suite['shader_args']['v2']
            for shader in suite['shader_args']['v1']:
                assert snapshots['v1'][shader] == snapshots['v2'][shader]
    # Verify binary linkage facts using the existing MIT read-only dumper.
    def sig(name, chunk):
        return signatures.signatures(here / 'v2/shaders' / (name + '.dxil'))['signatures'].get(chunk, [])
    assert not sig('solid_link', 'ISG1')
    def contract(elements):
        return [(e['semantic'], e['index'], e['register'], e['mask'], e['component_type']) for e in elements]
    assert contract(sig('tess_hs', 'PSG1')) == contract(sig('tess_ds', 'PSG1'))
    assert len(sig('tess_ds', 'PSG1')) == 6
    if args.git_origin:
        for rel, expected in snapshots['v1'].items():
            data = subprocess.check_output(['git', 'show', manifest['legacy_commit'] +
                 ':src/asahi/d3d12/tests/' + rel])
            assert hashlib.sha256(data).hexdigest() == expected, rel
        print('v1: byte-identical to legacy Git commit')
    print('Only the six declared fixtures differ; signatures match')


if __name__ == '__main__':
    main()

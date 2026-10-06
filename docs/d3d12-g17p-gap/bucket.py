#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Partition the accepted Neo strict B-only ledger by first observed mechanism.

This does not infer a GPU failure from a test name. Separate stderr/stdout
streams have no common timestamp: the first non-cascading stderr diagnostic is
the mechanism, and the first stdout assertion/skip is retained separately.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


RULES = [
    ('background/EOT producer unavailable', 'native graphics admission', ['P04-P08'], ['P14']),
    ('native BG/EOT rejected', 'render target/layout', [], ['P08', 'P14']),
    ('native depth render producer', 'depth admission', ['P02'], ['P14']),
    ('native texture compilation subset', 'texture compiler subset', ['P03'], ['P14']),
    ('typed-buffer stores require', 'typed buffer addressing', ['P03', 'P09'], ['P14', 'P15']),
    ('image atomics require', 'image atomics addressing', ['P03', 'P09'], ['P14', 'P15']),
    ('libagx program', 'native meta compute program', ['P09', 'P10-P17'], ['P15', 'P16']),
    ('native indirect/predicated compute', 'indirect compute admission', ['P17'], ['P10', 'P15']),
    ('native direct buffer-compute', 'compute ABI', ['P09'], ['P15']),
    ('resource range is not bound', 'descriptor/binding', ['P03'], ['P06', 'P12']),
    ('shader resource array crosses', 'descriptor/binding', ['P03'], ['P06', 'P12']),
    ('incompatible shader signatures', 'shader linkage', ['P10-P17'], ['P12', 'P16']),
    ('CopyTextureRegion', 'copy/layout', [], ['P08']),
]


def mechanism(stderr, stdout, row):
    lines = stderr.splitlines()
    # Unsupported optional QI probes and GetInterface are emitted at startup
    # for every Proton test. They are not the first failing mechanism.
    for i, line in enumerate(lines, 1):
        if 'QueryInterface:' in line or 'unimplemented: D3D12GetInterface' in line:
            continue
        if not any(word in line.lower() for word in ('error:', 'refus', 'reject', 'assertion')):
            continue
        for needle, bucket, phase, parity in RULES:
            if needle in line:
                return bucket, phase, parity, line, i, 'stderr'
        if 'recording into a non-recording' in line:
            # A cascading recording error cannot identify its original cause.
            continue
        return 'other diagnostic (unresolved)', [], [], line, i, 'stderr'
    for i, line in enumerate(stdout.splitlines(), 1):
        if 'Test skipped:' in line and row['A']['status'] == 'SKIP':
            return 'capability/API or adapter skip', [], ['P01', 'P02'], line, i, 'stdout'
    if row['A']['status'] in ('CRASH', 'TIMEOUT'):
        return 'crash/timeout without mechanism', [], [], row['A']['status'], None, None
    return 'assertion failure (unresolved)', [], [], None, None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('scoreboard', type=Path)
    ap.add_argument('out', type=Path)
    args = ap.parse_args()
    source = args.scoreboard / 'STATS.json'
    stats = json.loads(source.read_text())
    gap = [r for r in stats['tests'] if r['machine'] == 'neo' and
           r['B']['status'] == 'PASS' and r['A']['status'] != 'PASS']
    assert len(gap) == 496, f'Expected accepted 496, found {len(gap)}'
    assert len({(r['corpus'], r['test']) for r in gap}) == 496
    cache = {}

    def logs(backend):
        results = Path(backend['evidence'])
        if results not in cache:
            data = json.loads(results.read_text())
            cache[results] = data['results'] if isinstance(data, dict) else data
        item = next(r for r in cache[results] if r['test'] == row['test'])
        paths = {k: str(results.parent / item[k]) for k in ('stdout', 'stderr')}
        return item, paths

    rows = []
    for row in gap:
        a, paths = logs(row['A'])
        b, bpaths = logs(row['B'])
        assert a['status'] == row['A']['status'] and b['status'] == 'PASS'
        texts = {k: Path(v).read_text(errors='replace') for k, v in paths.items()}
        bucket, phase, parity, line, lineno, stream = mechanism(texts['stderr'], texts['stdout'], row)
        assertion = next((l for l in texts['stdout'].splitlines()
                          if 'Test failed:' in l or 'Test skipped:' in l), None)
        # Stage names are planning context, never evidence for the mechanism.
        stage = 'P10-P17 / parity P16' if any(s in row['test'].lower() for s in
                    ('geometry', 'tessellation', 'hull', 'domain', 'mesh', 'amplification')) else None
        macs = {m: next(r['A']['status'] for r in stats['tests'] if r['machine'] == m and
                       r['corpus'] == row['corpus'] and r['test'] == row['test'])
                for m in ('m1pro', 'm2')}
        rows.append(dict(corpus=row['corpus'], test=row['test'], file=row['file'],
                         bucket=bucket, A=row['A']['status'], B='PASS',
                         a_driver=row['A']['driver'], b_driver=row['B']['driver'],
                         first_mechanism=line, mechanism_line=lineno, mechanism_stream=stream,
                         first_assertion=assertion, phase2=phase, parity=parity,
                         stage_planning_context=stage, mac_A=macs,
                         A_logs=paths, B_logs=bpaths,
                         A_command=str(Path(paths['stdout']).with_suffix('.command.json')),
                         B_command=str(Path(bpaths['stdout']).with_suffix('.command.json')),
                         A_log_sha256={k: hashlib.sha256(Path(v).read_bytes()).hexdigest()
                                       for k, v in paths.items()}))
    counts = Counter(r['bucket'] for r in rows)
    args.out.mkdir(parents=True, exist_ok=True)
    ledger = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  policy='First observed non-cascading stderr mechanism; unresolved causes are explicit. '
                         'Separate streams do not establish cross-stream chronology.',
                  total=len(rows), buckets=dict(counts.most_common()), cases=rows)
    (args.out / 'BUCKETS.json').write_text(json.dumps(ledger, indent=2) + '\n')
    with (args.out / 'CASES.tsv').open('w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['corpus', 'test', 'A', 'B', 'bucket', 'phase2', 'parity', 'first_mechanism', 'A_stderr'])
        for r in rows:
            writer.writerow([r['corpus'], r['test'], r['A'], r['B'], r['bucket'],
                             ','.join(r['phase2']), ','.join(r['parity']),
                             r['first_mechanism'], r['A_logs']['stderr']])
    md = ['# Accepted Neo B-only gap', '', '| First observed mechanism | Cases |', '|---|---:|']
    md += [f'| {k} | {v} |' for k, v in counts.most_common()]
    md += ['', 'Total: 496. Full case names, A/B commands, log hashes, first assertions and plan slices '
           'are in BUCKETS.json and CASES.tsv. Counts describe the accepted 82dee6e1 measurement, '
           'not the newer fallback candidate.', '',
           'The first 291 graphics-admission refusals mask later vertex/index/instancing, format/layout, '
           'descriptor and stage failures. Reclassify only after an explicit physical-admission run. '
           'Zero direct observations in a category do not prove that producer works.']
    (args.out / 'BUCKETS.md').write_text('\n'.join(md) + '\n')
    # Exact inventory rows retain the pinned upstream inputs and categories.
    inventory = json.loads((args.scoreboard / 'inventory.json').read_text())
    for bucket in counts:
        selected = {(r['corpus'], r['test']) for r in rows if r['bucket'] == bucket}
        inv = [r for r in inventory if (r['corpus'], r['test']) in selected]
        assert len(inv) == len(selected)
        name = bucket.replace('/', '-').replace(' ', '-').replace('(', '').replace(')', '')
        (args.out / f'inventory-{name}.json').write_text(json.dumps(inv, indent=2) + '\n')
    print(json.dumps(dict(total=len(rows), buckets=counts), indent=2))


if __name__ == '__main__':
    main()

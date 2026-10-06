#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Publish individual runs without borrowing passes across driver revisions."""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('report', type=Path)
    p.add_argument('stats', type=Path)
    args = p.parse_args()
    args.stats.mkdir(parents=True, exist_ok=True)
    summaries = []
    for path in sorted((args.report / 'raw').glob('*/results.json')):
        data = json.loads(path.read_text())
        identity = data['identity']
        frozen = identity.get('corpus') is None
        complete = data.get('attempted') == 17 if frozen else (path.parent / 'COMPLETE').exists()
        validity = 'complete' if complete else 'incomplete'
        if identity['machine'] == 'neo':
            before = path.parent / 'journal-before.log'
            after = path.parent / 'journal-after.log'
            clean = before.exists() and after.exists() and all(
                'submit failed' not in f.read_text(errors='replace') for f in (before, after))
            if not clean or (path.parent / 'INVALID-KERNEL').exists():
                validity = 'invalid kernel evidence'
        summary = dict(task='t-1189', machine=identity['machine'], driver=identity['driver'],
                       corpus=identity.get('corpus', 'frozen17'), evidence=str(path),
                       attempted=len(data['results']), strict=dict(Counter(
                           row['status'] for row in data['results'])), validity=validity,
                       identity=identity, results=data['results'])
        (args.stats / f't1189-{path.parent.name}.json').write_text(json.dumps(summary, indent=2) + '\n')
        summaries.append({k: v for k, v in summary.items() if k not in ('identity', 'results')})
    # A refused preflight is not a measured FAIL or an empty successful run.
    for journal in sorted((args.report / 'raw').glob('i2-*/journal-before.log')):
        if (journal.parent / 'results.json').exists():
            continue
        count = sum('submit failed' in line for line in journal.read_text(errors='replace').splitlines())
        if not count:
            continue
        name = journal.parent.name
        corpus = ('vkd3d-proton-tests' if 'proton' in name else
                  'vkd3d-shader-tests' if 'shader' in name else 'vkd3d-tests')
        driver = 'd3d12agx@3b1710ace113' if 'base' in name else 'd3d12agx@8f7347b64e6'
        summary = dict(task='t-1189', machine='neo', driver=driver, corpus=corpus,
                       evidence=str(journal), attempted=0, strict={},
                       validity=f'NOT_RUN: preflight submit-failure count {count}')
        (args.stats / f't1189-{name}.json').write_text(json.dumps(summary, indent=2) + '\n')
        summaries.append(summary)
    (args.report / 'RUN-STATS.json').write_text(json.dumps(summaries, indent=2) + '\n')
    lines = ['# Individual measured runs', '',
             'Every row stands alone; historical passes never fill a newer unmeasured cell.', '',
             '| Run | Machine | Driver | Attempted | Strict outcomes | Validity |', '|---|---|---|---:|---|---|']
    for row in summaries:
        name = Path(row['evidence']).parent.name
        counts = ', '.join(f'{key} {value}' for key, value in row['strict'].items())
        lines.append(f'| {name} | {row["machine"]} | {row["driver"]} | {row["attempted"]} | {counts} | {row["validity"]} |')
    (args.report / 'RUN-STATS.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()

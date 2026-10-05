#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run one upstream registration per process; preserve exact commands and logs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import time


SUMMARY = re.compile(r'(\d+) tests executed \((\d+) failures, (\d+) successful todo, (\d+) skipped(?:, (\d+) todo)?(?:, (\d+) bugs)?\)')
WINE_SUMMARY = re.compile(r'(\d+) tests executed \((\d+) failures, (\d+) skipped, (\d+) todo, (\d+) bugs\)')


def classify(rc, output, timed_out=False):
    counts = [tuple(int(x or 0) for x in m) for m in SUMMARY.findall(output)]
    counts += [(int(n), int(f), 0, int(s), int(t), int(b))
               for n, f, s, t, b in WINE_SUMMARY.findall(output)]
    if timed_out:
        return 'TIMEOUT', counts
    if rc < 0 or rc in (134, 139):
        return 'CRASH', counts
    if rc or any(c[1] or c[4] or c[5] for c in counts):
        return 'FAIL', counts
    if not counts:
        return 'NO_VERDICT', counts
    if any(c[3] for c in counts) or not sum(c[0] for c in counts):
        return 'SKIP', counts
    return 'PASS', counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--corpus', required=True)
    p.add_argument('--machine', required=True)
    p.add_argument('--driver', required=True)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--src', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--timeout', type=float, default=20)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--limit', type=int)
    p.add_argument('--wall-budget', type=float, help='Stop between tests to keep hardware requests short')
    p.add_argument('--library-path', default='')
    p.add_argument('--env', action='append', default=[], metavar='NAME=VALUE',
                   help='Explicit driver opt-in, recorded in every command and identity')
    p.add_argument('--icd', help='Explicit Honeykrisp ICD JSON; prevents software fallback')
    args = p.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    rows = [r for r in json.loads(args.inventory.read_text()) if r['corpus'] == args.corpus]
    rows = rows[args.start:args.start + args.limit if args.limit is not None else None]
    args.out.mkdir(parents=True, exist_ok=True)
    # Do not silently replace a prior result batch.
    if (args.out / 'results.json').exists():
        raise SystemExit('Output already contains results; choose a new batch directory')
    env = {k: os.environ[k] for k in ('PATH', 'HOME', 'LANG', 'LC_ALL', 'TMPDIR')
           if k in os.environ}
    env.update(VKD3D_TEST_DEBUG='1', VKD3D_DEBUG='warn')
    for item in args.env:
        name, sep, value = item.partition('=')
        if not sep or not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', name):
            p.error('--env requires NAME=VALUE')
        env[name] = value
    if args.library_path:
        env['LD_LIBRARY_PATH'] = args.library_path
    if args.icd:
        env['VK_DRIVER_FILES'] = args.icd
    identity = dict(time_utc=datetime.now(timezone.utc).isoformat(), machine=args.machine,
                    driver=args.driver, corpus=args.corpus, binary=str(args.binary.resolve()),
                    binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(), env=env,
                    inventory_sha256=hashlib.sha256(args.inventory.read_bytes()).hexdigest())
    (args.out / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    with (args.out / 'uname.txt').open('w') as output:
        subprocess.run(['uname', '-a'], stdout=output, stderr=subprocess.STDOUT, check=False)
    with (args.out / 'kernel-before.log').open('w') as output:
        subprocess.run(['dmesg', '--color=never'], stdout=output, stderr=subprocess.STDOUT, check=False)
    subprocess.run(['ldd', str(args.binary.resolve())], stdout=(args.out / 'ldd.txt').open('w'),
                   stderr=subprocess.STDOUT, env=env, check=False)
    results = []
    batch_start = time.monotonic()
    for i, row in enumerate(rows):
        # Leave a whole per-test timeout inside the budget; a budget cut-off
        # must not masquerade as a test timeout or a skipped registration.
        if args.wall_budget and time.monotonic() - batch_start + args.timeout > args.wall_budget:
            break
        stem = f'{i + args.start:04d}'
        cmd = [str(args.binary.resolve())]
        test_env = env.copy()
        if args.corpus == 'vkd3d-shader-tests':
            cmd += ['--filter-executor', 'vkd3d', str((args.src / row['input']).resolve())]
        else:
            # Proton has exact-match filtering. WineHQ 1.17 uses substring filtering;
            # its adapter patch adds the same exact-match filter (see adapt.py).
            test_env['VKD3D_TEST_MATCH'] = row['test']
        (args.out / f'{stem}.command.json').write_text(json.dumps(dict(argv=cmd, env=test_env), indent=2) + '\n')
        start = time.monotonic()
        timed_out = False
        with (args.out / f'{stem}.stdout').open('wb') as out, (args.out / f'{stem}.stderr').open('wb') as err:
            proc = subprocess.Popen(cmd, stdout=out, stderr=err, env=test_env, start_new_session=True)
            try:
                rc = proc.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(proc.pid, signal.SIGKILL)
                rc = proc.wait()
        output = (args.out / f'{stem}.stdout').read_text(errors='replace')
        status, counts = classify(rc, output, timed_out)
        # A broken device discovery must not turn an entire corpus into feature skips.
        results.append(dict(row, status=status, rc=rc, seconds=round(time.monotonic() - start, 3),
                            counts=counts, stdout=f'{stem}.stdout', stderr=f'{stem}.stderr'))
        (args.out / f'{stem}.rc').write_text(str(rc) + '\n')
        temporary = args.out / 'results.json.tmp'
        temporary.write_text(json.dumps(dict(identity=identity, results=results), indent=2) + '\n')
        temporary.replace(args.out / 'results.json')
        print(f'{args.machine} {args.driver} {row["test"]}: {status}', flush=True)
    with (args.out / 'kernel-after.log').open('w') as output:
        subprocess.run(['dmesg', '--color=never'], stdout=output, stderr=subprocess.STDOUT, check=False)


if __name__ == '__main__':
    main()

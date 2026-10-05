#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""t-1083 run-v8.py with explicit bounded baseline flags; no GS admission override.
Original SHA256: 4e4e75a6b576c2754fe96cd9d76cf32fea7b132e0efdc90b3c13450ff1f877d9
"""
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


def memory_cgroup():
    try:
        group = next(line.split(':', 2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines()
                     if line.startswith('0::'))
        directory = Path('/sys/fs/cgroup') / group.lstrip('/')
        limits = {name: (directory / name).read_text().strip()
                  for name in ('memory.max', 'memory.swap.max')}
        events = dict(line.split() for line in (directory / 'memory.events').read_text().splitlines())
        return dict(path=str(directory), limits=limits, oom_kill=int(events.get('oom_kill', 0)))
    except (OSError, StopIteration, ValueError):
        return {}


def test_oom_score():
    # A resource-exhaustion test should be the OOM victim, rather than the
    # small supervisor that must save its result. Set only this child process.
    try:
        Path('/proc/self/oom_score_adj').write_text('0')
    except OSError:
        pass


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
    p.add_argument('--icd', help='Explicit Honeykrisp ICD JSON; prevents software fallback')
    p.add_argument('--physical-baseline', action='store_true', help='Existing bounded native baseline and D32 admission only')
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
    if args.physical_baseline:
        env.update(AURORA_D3D12_PHYSICAL_BASELINE='1', AURORA_D3D12_PHYSICAL_D32='1')
    if args.library_path:
        env['LD_LIBRARY_PATH'] = args.library_path
    if args.icd:
        env['VK_DRIVER_FILES'] = args.icd
    identity = dict(time_utc=datetime.now(timezone.utc).isoformat(), machine=args.machine,
                    driver=args.driver, corpus=args.corpus, binary=str(args.binary.resolve()),
                    binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(), env=env,
                    inventory_sha256=hashlib.sha256(args.inventory.read_bytes()).hexdigest(),
                    memory_cgroup=memory_cgroup())
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
        unreaped = False
        oom_before = memory_cgroup().get('oom_kill', 0)
        with (args.out / f'{stem}.stdout').open('wb') as out, (args.out / f'{stem}.stderr').open('wb') as err:
            proc = subprocess.Popen(cmd, stdout=out, stderr=err, env=test_env,
                                    start_new_session=True, preexec_fn=test_oom_score)
            try:
                rc = proc.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(proc.pid, signal.SIGKILL)
                try:
                    rc = proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # An uninterruptible kernel wait cannot be fixed by sending
                    # more signals. Record the incomplete test and stop this batch.
                    rc = None
                    unreaped = True
        stdout_path = args.out / f'{stem}.stdout'
        # The runner prints its summary last. Failure storms can produce large
        # logs; preserve them whole, but parse only a bounded final window.
        with stdout_path.open('rb') as log:
            log.seek(max(0, stdout_path.stat().st_size - 65536))
            output = log.read().decode(errors='replace')
        status, counts = classify(rc, output, timed_out)
        # A broken device discovery must not turn an entire corpus into feature skips.
        results.append(dict(row, status=status, rc=rc, seconds=round(time.monotonic() - start, 3),
                            counts=counts, stdout=f'{stem}.stdout', stderr=f'{stem}.stderr',
                            unreaped=unreaped, pid=proc.pid if unreaped else None,
                            cgroup_oom_kills=max(0, memory_cgroup().get('oom_kill', 0) - oom_before)))
        (args.out / f'{stem}.rc').write_text(str(rc) + '\n')
        temporary = args.out / 'results.json.tmp'
        temporary.write_text(json.dumps(dict(identity=identity, results=results), indent=2) + '\n')
        temporary.replace(args.out / 'results.json')
        print(f'{args.machine} {args.driver} {row["test"]}: {status}', flush=True)
        if unreaped:
            print('Batch stopped: killed test remains in a kernel wait; queue recovery is required.', flush=True)
            break
    with (args.out / 'kernel-after.log').open('w') as output:
        subprocess.run(['dmesg', '--color=never'], stdout=output, stderr=subprocess.STDOUT, check=False)


if __name__ == '__main__':
    main()

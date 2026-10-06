#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the unchanged frozen ba4c runner with bounded, logging wrappers."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def clean_submits(out, phase):
    log = subprocess.run(['journalctl', '-k', '-b', '--no-pager', '-o', 'cat'],
                         capture_output=True, text=True)
    (out / f'journal-{phase}.log').write_text(log.stdout + log.stderr)
    if log.returncode:
        raise RuntimeError('Cannot verify kernel submit failures')
    count = sum('submit failed' in line for line in log.stdout.splitlines())
    print(f'kernel submit failed count ({phase}): {count}', flush=True)
    return count == 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('prefix', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--library-dir', type=Path)
    p.add_argument('--physical', action='store_true')
    p.add_argument('--driver', required=True)
    p.add_argument('--machine', required=True)
    args = p.parse_args()
    prefix = args.prefix.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    tests = prefix / 'reference/src/asahi/d3d12/tests'
    reference = prefix / 'reference'
    if args.machine == 'neo' and not clean_submits(out, 'before'):
        raise SystemExit('Invalid GPU environment: this boot has submit failures')
    for line in (prefix / 'reference-inputs.sha256').read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        name = name.lstrip('*')
        assert hashlib.sha256((reference / name).read_bytes()).hexdigest() == expected, name
    env = {k: os.environ[k] for k in ('PATH', 'HOME', 'LANG', 'LC_ALL', 'TMPDIR') if k in os.environ}
    library = (args.library_dir or prefix / 'bin').resolve()
    env['LD_LIBRARY_PATH'] = str(library)
    if args.physical:
        env.update(AURORA_D3D12_PHYSICAL_BASELINE='1', AURORA_D3D12_PHYSICAL_D32='1')
    identity = dict(time_utc=datetime.now(timezone.utc).isoformat(), machine=args.machine,
                    driver=args.driver, env=env, reference='ba4cfd5dfa4c4d6f5b88adcf9ac5eb233c3442a4',
                    library_sha256=hashlib.sha256((library / 'libd3d12agx.so').read_bytes()).hexdigest())
    if args.machine == 'neo':
        identity['boot_id'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    (out / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    wrappers = out / 'wrappers'
    wrappers.mkdir()
    wrapper = '''#!/usr/bin/env python3
import json, os, pathlib, signal, subprocess, sys
out=pathlib.Path(os.environ['FROZEN_OUT'])
name=pathlib.Path(sys.argv[0]).name
stem=name.removeprefix('d3d12agx_')
cmd=[str(pathlib.Path(os.environ['FROZEN_BIN'])/name), *sys.argv[1:]]
(out/(stem+'.command.json')).write_text(json.dumps(dict(argv=cmd, env=dict(os.environ)),indent=2))
with (out/(stem+'.stdout')).open('wb') as stdout, (out/(stem+'.stderr')).open('wb') as stderr:
    proc=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,start_new_session=True)
    try: rc=proc.wait(timeout=25)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGKILL)
        proc.wait(timeout=5)
        rc=124
(out/(stem+'.rc')).write_text(str(rc)+'\\n')
sys.stdout.write((out/(stem+'.stdout')).read_text(errors='replace'))
sys.stderr.write((out/(stem+'.stderr')).read_text(errors='replace'))
sys.exit(rc if rc>=0 else 128-rc)
'''
    for exe in sorted((prefix / 'bin').glob('d3d12agx_g*')):
        path = wrappers / exe.name
        path.write_text(wrapper)
        path.chmod(0o755)
    env.update(FROZEN_OUT=str(out), FROZEN_BIN=str(prefix / 'bin'))
    command = ['bash', str(tests / 'run-suites.sh'), str(wrappers)]
    (out / 'command.json').write_text(json.dumps(dict(argv=command, env=env), indent=2) + '\n')
    with (out / 'TEST.log').open('wb') as log:
        rc = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    rows = []
    for path in sorted(out.glob('*.rc')):
        name = path.stem
        stdout = (out / f'{name}.stdout').read_text(errors='replace')
        stderr = (out / f'{name}.stderr').read_text(errors='replace')
        child_rc = int(path.read_text())
        passed = child_rc == 0 and 'ALL PASS' in stdout + stderr and not re.search(
            r'^FAIL|BLOCKED', stdout + '\n' + stderr, re.M)
        rows.append(dict(test=name, status='PASS' if passed else 'FAIL', rc=child_rc))
    result = dict(identity=identity, results=rows, attempted=len(rows),
                  passes=sum(r['status'] == 'PASS' for r in rows), runner_rc=rc)
    result['valid_kernel'] = args.machine != 'neo' or clean_submits(out, 'after')
    if not result['valid_kernel']:
        (out / 'INVALID-KERNEL').write_text('Submit failures invalidate this frozen17 run.\n')
    (out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(attempted=result['attempted'], passes=result['passes'], runner_rc=rc)))
    assert len(rows) == 17, 'Frozen runner did not attempt all 17 suites'


if __name__ == '__main__':
    main()

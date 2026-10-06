#!/usr/bin/env python3
# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT
"""Run a bounded Neo GPU test inside an approved aurora-ctl execute request.

This is an on-target test wrapper, not a transport. Upload it with aurora-ctl,
then request execute: python3 neo-gpu-session.py EVIDENCE_DIR -- COMMAND ARGS.
It refuses a boot with submit failures or fewer than 40 free firmware QIDs,
and stops the command at the first new quarantine, poison or submit failure.
"""

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time


QUEUE = Path('/sys/module/asahi/parameters/g17p_queue_slots')
BOOT = Path('/proc/sys/kernel/random/boot_id')
BAD = re.compile(r'QUARANTINE|POISONED|submit failed')


def journal(cursor=None):
    args = ['journalctl', '-k', '-b', '--no-pager', '-o', 'short-monotonic',
            '--show-cursor']
    if cursor:
        args += ['--after-cursor', cursor]
    out = subprocess.check_output(args, text=True, stderr=subprocess.PIPE)
    match = re.search(r'^-- cursor: (.+)$', out, re.MULTILINE)
    if not match:
        raise RuntimeError('kernel journal has no cursor')
    return out[:match.start()], match[1]


def descendants(root, known):
    # Proton's runner uses setsid. Track its descendants' groups as well as
    # the wrapper's group so a stop also reaches those detached children.
    procs = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            procs[int(path.parent.name)] = (int(fields[1]), int(fields[2]))
        except (OSError, ValueError, IndexError):
            continue
    owned = {root}
    while True:
        more = {pid for pid, (parent, group) in procs.items()
                if parent in owned or group in known}
        if more <= owned:
            break
        owned |= more
    known.update(procs[pid][1] for pid in owned if pid in procs)
    known.discard(os.getpgrp())


def stop(child, groups):
    descendants(child.pid, groups)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if child.poll() is None:
            child.send_signal(sig)
        for group in groups:
            try:
                os.killpg(group, sig)
            except ProcessLookupError:
                pass
            except PermissionError:
                # Some host test environments allow signaling a child but
                # prohibit process-group signals. Neo sessions run as root.
                if os.uname().sysname == 'Linux':
                    raise
        if sig == signal.SIGTERM:
            time.sleep(2)
    child.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence', type=Path)
    parser.add_argument('--timeout', type=float, default=540)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ['--']:
        command = command[1:]
    if not command:
        parser.error('a test command is required')
    if not 0 < args.timeout <= 540:
        parser.error('timeout must be in (0, 540] seconds')
    args.evidence.mkdir(parents=True, exist_ok=False)
    result = dict(command=command, valid=False, test_rc=None)
    child = None
    groups = set()
    started = time.monotonic()
    try:
        result['boot_id'] = BOOT.read_text().strip()
        result['kernel'] = os.uname().release
        queue = QUEUE.read_text()
        (args.evidence / 'queue-before.txt').write_text(queue)
        before, cursor = journal()
        (args.evidence / 'kernel-before.txt').write_text(before)
        result['submit_failed_before'] = before.count('submit failed')
        match = re.search(r'^qids .*?\bfree=(\d+)', queue)
        if not match or int(match[1]) < 40:
            raise RuntimeError('fewer than 40 free QIDs, or unknown QID budget; request reboot from orchestrator')
        if result['submit_failed_before']:
            raise RuntimeError('boot already has submit failures; GPU result would be invalid')
        with (args.evidence / 'command.log').open('w') as log, \
                (args.evidence / 'kernel-new.txt').open('w') as updates:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            groups.add(child.pid)
            while True:
                descendants(child.pid, groups)
                new, cursor = journal(cursor)
                updates.write(new)
                updates.flush()
                if BAD.search(new):
                    raise RuntimeError('first kernel quarantine, poison or submit failure; session stopped')
                if child.poll() is not None:
                    result['test_rc'] = child.returncode
                    break
                if time.monotonic() - started >= args.timeout:
                    raise RuntimeError('bounded GPU session timed out')
                time.sleep(0.25)
        after, _ = journal()
        result['submit_failed_after'] = after.count('submit failed')
        if result['submit_failed_after'] or BOOT.read_text().strip() != result['boot_id']:
            raise RuntimeError('submit count rose or boot changed; GPU result invalid')
        result['valid'] = True
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        result['error'] = str(exc)
        if child is not None:
            stop(child, groups)
            result['test_rc'] = child.returncode
    finally:
        result['elapsed_s'] = round(time.monotonic() - started, 3)
        (args.evidence / 'queue-after.txt').write_text(QUEUE.read_text())
        after, _ = journal()
        (args.evidence / 'kernel-after.txt').write_text(after)
        result['submit_failed_after'] = after.count('submit failed')
        (args.evidence / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    return (result['test_rc'] or 0) if result['valid'] else 125


if __name__ == '__main__':
    raise SystemExit(main())

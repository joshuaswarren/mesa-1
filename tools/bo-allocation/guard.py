#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run one bounded measurement and stop at the first GPU quarantine."""
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=False)
slots = Path('/sys/module/asahi/parameters/g17p_queue_slots')
boot = Path('/proc/sys/kernel/random/boot_id').read_text()
(out / 'boot-id').write_text(boot)


def capture(name):
    log = subprocess.check_output(['journalctl', '-k', '-b', '--no-pager'], text=True)
    (out / ('kernel-' + name)).write_text(log)
    queue = slots.read_text()
    (out / ('queue-' + name)).write_text(queue)
    return log, queue


before, queue = capture('before')
if re.search(r'submit failed.*ENOSPC', before):
    sys.exit('invalid boot: submit ENOSPC')
if int(re.search(r'free=(\d+)', queue)[1]) < 40:
    sys.exit('queue reserve below 40: reboot required')
start = time.time()
with (out / 'stdout').open('w') as stdout, (out / 'stderr').open('w') as stderr:
    child = subprocess.Popen(sys.argv[2:], start_new_session=True, stdout=stdout, stderr=stderr)
    reason = None
    while child.poll() is None:
        log = subprocess.check_output(['journalctl', '-k', '--since', '@' + str(int(start)),
                                       '--no-pager'], text=True)
        if re.search(r'QUARANTINE|POISONED|submit failed', log):
            reason = 'GPU health changed'
        if time.time() - start > 550:
            reason = 'session time limit'
        if reason:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            break
        time.sleep(0.25)
after, queue = capture('after')
valid = (boot == Path('/proc/sys/kernel/random/boot_id').read_text()
         and before.count('submit failed') == after.count('submit failed')
         and not re.search(r'submit failed.*ENOSPC', after) and reason is None)
(out / 'result.json').write_text(json.dumps(dict(rc=child.returncode, valid=valid,
    reason=reason, duration_s=time.time() - start,
    submit_before=before.count('submit failed'), submit_after=after.count('submit failed')), indent=2))
sys.exit(child.returncode if valid else 3)

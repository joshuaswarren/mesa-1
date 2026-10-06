# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT
"""CPU-only checks of the GPU session admission and first-failure stop."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    'session', Path(__file__).with_name('neo-gpu-session.py'))
session = importlib.util.module_from_spec(spec)
spec.loader.exec_module(session)


class SessionTests(unittest.TestCase):
    def run_session(self, before='', updates='', free=100, command=None,
                    timeout=180):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue, boot = root / 'queue', root / 'boot'
            queue.write_text(f'qids leased=28 free={free} published=28\n')
            boot.write_text('test-boot\n')
            evidence = root / 'evidence'
            marker = root / 'ran'
            # A marker proves that a refused preflight never started the test.
            command = command or [sys.executable, '-c',
                                  'from pathlib import Path; import sys; '
                                  'Path(sys.argv[1]).touch()', str(marker)]
            def journal(cursor=None):
                return (updates if cursor else before), 'test-cursor'
            with patch.object(session, 'QUEUE', queue), \
                    patch.object(session, 'BOOT', boot), \
                    patch.object(session, 'journal', journal), \
                    patch.object(sys, 'argv', ['session', '--timeout', str(timeout),
                                              str(evidence), '--', *command]), \
                    contextlib.redirect_stdout(io.StringIO()):
                rc = session.main()
            return rc, json.loads((evidence / 'result.json').read_text()), marker.exists()

    def test_clean_session(self):
        rc, result, ran = self.run_session()
        self.assertEqual(rc, 0)
        self.assertTrue(result['valid'])
        self.assertTrue(ran)
        self.assertEqual(result['submit_failed_after'], 0)

    def test_contaminated_boot_never_starts(self):
        for error in ('ENOSPC', 'ENOENT'):
            with self.subTest(error=error):
                rc, result, ran = self.run_session(before=f'submit failed {error}\n')
                self.assertEqual(rc, 125)
                self.assertFalse(result['valid'])
                self.assertFalse(ran)

    def test_low_queue_budget_never_starts(self):
        rc, result, ran = self.run_session(free=39)
        self.assertEqual(rc, 125)
        self.assertIn('request reboot', result['error'])
        self.assertFalse(ran)

    def test_first_kernel_failure_stops_child(self):
        for failure in ('QUARANTINE', 'POISONED', 'submit failed ENOSPC'):
            with self.subTest(failure=failure):
                rc, result, _ = self.run_session(
                    updates=failure + '\n', command=[sys.executable, '-c',
                                                    'import time; time.sleep(60)'])
                self.assertEqual(rc, 125)
                self.assertFalse(result['valid'])
                self.assertLess(result['test_rc'], 0)
                self.assertLess(result['elapsed_s'], 10)

    def test_deadline_stops_child_without_kernel_error(self):
        rc, result, _ = self.run_session(
            timeout=0.1, command=[sys.executable, '-c',
                                 'import time; time.sleep(60)'])
        self.assertEqual(rc, 125)
        self.assertIn('timed out', result['error'])
        self.assertLess(result['test_rc'], 0)
        self.assertLess(result['elapsed_s'], 10)


if __name__ == '__main__':
    unittest.main()

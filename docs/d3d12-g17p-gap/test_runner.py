#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Host tests for explicit admission and strict verdict accounting."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('gap_runner', HERE / 'run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def test_strict_verdicts(self):
        good = '1 tests executed (0 failures, 0 successful todo, 0 skipped)'
        self.assertEqual(runner.classify(0, good)[0], 'PASS')
        self.assertEqual(runner.classify(0, good.replace('0 skipped', '1 skipped'))[0], 'SKIP')
        self.assertEqual(runner.classify(0, '1 tests executed (0 failures, 0 skipped, 1 todo, 0 bugs)')[0], 'FAIL')
        self.assertEqual(runner.classify(-11, good)[0], 'CRASH')
        self.assertEqual(runner.classify(0, good, True)[0], 'TIMEOUT')
        self.assertEqual(runner.classify(0, '')[0], 'NO_VERDICT')

    def test_explicit_environment(self):
        # Keep host scratch in the assigned S500 worktree.
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root = Path(tmp)
            fake = root / 'fake-test'
            fake.write_text(f'#!{sys.executable}\nprint("1 tests executed (0 failures, 0 successful todo, 0 skipped)")\n')
            fake.chmod(0o755)
            for name in ('ldd', 'uname', 'dmesg'):
                path = root / name
                path.write_text('#!/bin/sh\nexit 0\n')
                path.chmod(0o755)
            inventory = root / 'inventory.json'
            inventory.write_text(json.dumps([dict(corpus='vkd3d-tests', test='test_fake')]))
            env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ['PATH'],
                       AURORA_D3D12_PHYSICAL_BASELINE='1', AURORA_D3D12_PHYSICAL_D32='1')
            for flags, expected in (([], {}), (['--physical-baseline'],
                  {'AURORA_D3D12_PHYSICAL_BASELINE': '1'}),
                  (['--physical-baseline', '--physical-d32'],
                  {'AURORA_D3D12_PHYSICAL_BASELINE': '1', 'AURORA_D3D12_PHYSICAL_D32': '1'})):
                out = root / ('out' + str(len(flags)))
                cmd = [sys.executable, str(HERE / 'run.py'), '--inventory', str(inventory),
                       '--src', str(root), '--machine', 'host', '--corpus', 'vkd3d-tests',
                       '--driver', 'fixture', '--binary', str(fake), '--out', str(out), *flags]
                subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL)
                identity = json.loads((out / 'identity.json').read_text())
                recorded = {k: v for k, v in identity['env'].items() if k.startswith('AURORA_')}
                self.assertEqual(recorded, expected)
                self.assertTrue((out / 'COMPLETE').exists())
                command = json.loads((out / '0000.command.json').read_text())
                self.assertEqual(command['env'], identity['env'] | {'VKD3D_TEST_MATCH': 'test_fake'})
                self.assertEqual(json.loads((out / 'results.json').read_text())['results'][0]['status'], 'PASS')
            result = subprocess.run([*cmd[:-2], '--physical-d32'],
                                    capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b'--physical-d32 requires --physical-baseline', result.stderr)

    def test_kernel_validity(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            out = Path(tmp)
            with patch.object(runner.subprocess, 'run', return_value=subprocess.CompletedProcess(
                    [], 0, stdout='ordinary message\n', stderr='')):
                self.assertTrue(runner.clean_submits(out, 'before'))
            with patch.object(runner.subprocess, 'run', return_value=subprocess.CompletedProcess(
                    [], 0, stdout='asahi submit failed: -28\n', stderr='')):
                self.assertFalse(runner.clean_submits(out, 'after'))
            self.assertIn('submit failed', (out / 'journal-after.log').read_text())
            with patch.object(runner.subprocess, 'run', return_value=subprocess.CompletedProcess(
                    [], 1, stdout='', stderr='Permission denied')):
                with self.assertRaises(RuntimeError):
                    runner.clean_submits(out, 'before')


if __name__ == '__main__':
    unittest.main()

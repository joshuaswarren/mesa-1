# SPDX-License-Identifier: MIT
# Copyright 2026 Ryan Murray <ryan@aurorasilicon.org>

import importlib.util
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location(
    "hk_gpu_run_validity", Path(__file__).resolve().parents[1] /
    "hk_gpu_run_validity.py"
)
VALIDITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDITY)

EIO = "asahi: IOCTL: submit failed! (submission ID: 42 err: EIO)\n"


class GpuRunValidityTest(unittest.TestCase):
    def test_preexisting_eio_is_allowed(self):
        before = EIO * 8 + "asahi: old QUARANTINE\n"
        result = VALIDITY.evaluate(before, before + "asahi: clean work\n")
        self.assertTrue(result["valid"])
        self.assertEqual(result["submit_failed_before"], 8)
        self.assertEqual(result["submit_failed_after"], 8)

    def test_new_submit_failure_is_invalid(self):
        self.assertFalse(VALIDITY.evaluate(EIO, EIO * 2)["valid"])

    def test_enospc_invalid_even_when_preexisting(self):
        for error in ("ENOSPC", "-28", "28"):
            log = "asahi: submit failed! err: " + error + "\n"
            with self.subTest(error=error):
                self.assertFalse(VALIDITY.evaluate(log, log)["valid"])
                self.assertFalse(VALIDITY.evaluate("", log)["valid"])

    def test_non_submit_enospc_does_not_count(self):
        result = VALIDITY.evaluate("", "other subsystem: ENOSPC\n")
        self.assertTrue(result["valid"])
        self.assertEqual(result["submit_enospc_after"], 0)


if __name__ == "__main__":
    unittest.main()

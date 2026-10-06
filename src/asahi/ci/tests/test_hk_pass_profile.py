# SPDX-License-Identifier: MIT
# Copyright 2026 Ryan Murray <ryan@aurorasilicon.org>

import importlib.util
import io
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location(
    "hk_pass_profile", Path(__file__).resolve().parents[1] /
    "summarize_hk_pass_profile.py"
)
PROFILE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROFILE)

LOG = """# HK_PASS_PROFILE pid=42 window_ms=2000
== window t=10.000 s=2.000 cmds=6 gpu_busy_ms=30.0 dropped=0 overflowed=0
    20.00 ms busy    24.00 ms span     4 x     8 draws  VDM present
    10.00 ms busy    10.00 ms span     2 x     4 draws  CDM | hk_xfer_copyimage2/pc57
== window t=12.000 s=2.000 cmds=4 gpu_busy_ms=16.0 dropped=1 overflowed=2
    16.00 ms busy    18.00 ms span     4 x     4 draws  VDM present
"""


class PassProfileTest(unittest.TestCase):
    def test_windows_include_first_and_last_rows(self):
        windows = list(PROFILE.windows(io.StringIO(LOG)))
        self.assertEqual([len(w["rows"]) for w in windows], [2, 1])
        self.assertEqual(windows[0]["rows"][1][0],
                         "CDM | hk_xfer_copyimage2/pc57")

    def test_aggregate_with_exact_present_count(self):
        result = PROFILE.summarize(io.StringIO(LOG), frame_key="VDM present")
        self.assertEqual(result["frames"], 8)
        self.assertEqual(result["fps"], 2)
        self.assertEqual(result["gpu_busy_ms_per_frame"], 46 / 8)
        self.assertEqual(result["passes"][0]["busy_ms_per_frame"], 36 / 8)
        self.assertEqual(result["passes"][1]["commands_per_frame"], 2 / 8)
        self.assertEqual((result["dropped"], result["overflowed"]), (1, 2))

    def test_interval_keeps_only_whole_windows(self):
        result = PROFILE.summarize(io.StringIO(LOG), start=11, end=14,
                                   frames=4)
        self.assertEqual(result["windows"], 1)
        self.assertEqual(result["gpu_busy_ms_per_frame"], 4)
        with self.assertRaisesRegex(ValueError, "no complete windows"):
            PROFILE.summarize(io.StringIO(LOG), start=11, end=13, frames=4)

    def test_reject_invalid_normalization(self):
        for options in ({}, {"frames": 0}, {"frames": -1},
                        {"frame_key": "absent"},
                        {"frames": 4, "frame_key": "VDM present"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                PROFILE.summarize(io.StringIO(LOG), **options)


if __name__ == "__main__":
    unittest.main()

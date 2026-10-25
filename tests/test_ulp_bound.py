"""CPU controls for the numerical GPU oracle, without starting Vulkan."""
import ast
from pathlib import Path
import unittest

import numpy as np


# ulp_bound currently dispatches a GPU job at module scope. Extract only its
# pure checks so these controls need neither a GPU nor a Vulkan driver.
source = Path(__file__).with_name("ulp_bound.py")
tree = ast.parse(source.read_text())
functions = ast.Module(body=[n for n in tree.body
                            if isinstance(n, ast.FunctionDef)
                            and n.name in ("quotient", "check")], type_ignores=[])
namespace = {"np": np}
exec(compile(functions, str(source), "exec"), namespace)


class ULPControls(unittest.TestCase):
    def measure(self, reference, result):
        namespace["inp"] = np.array([[reference, 0x3f800000]], dtype=np.uint32)
        namespace["out"] = np.array([[result]], dtype=np.uint32)
        return namespace["check"](0, namespace["inp"][:, 0],
                                  namespace["inp"][:, 1])

    def test_positive_one_ulp(self):
        r = self.measure(0x3f800000, 0x3f800001)
        self.assertEqual(r["max_ulp"], 1.0)
        self.assertEqual(r["over_2p5"], 0)

    def test_positive_over_budget(self):
        self.assertEqual(self.measure(0x3f800000, 0x3f800003)["over_2p5"], 1)

    def test_negative_errors_are_counted(self):
        r = self.measure(0xbf800000, 0xbf800003)
        self.assertEqual(r["max_ulp"], 3.0)
        self.assertEqual(r["over_2p5"], 1)

    def test_negative_power_of_two_boundary(self):
        self.assertEqual(self.measure(0xbf800000, 0xbf7fffff)["max_ulp"], 0.5)

    def test_nan_and_infinity_cannot_match_finite(self):
        for bits in (0x7fc00000, 0x7f800000, 0xff800000):
            with self.subTest(bits=hex(bits)):
                r = self.measure(0x3f800000, bits)
                self.assertEqual(r["over_2p5"], 1)
                self.assertEqual(r["nonfinite_mismatch"], 1)

    def test_largest_finite_has_finite_ulp(self):
        r = self.measure(0x7f7fffff, 0x7f7ffffe)
        self.assertEqual(r["max_ulp"], 1.0)
        self.assertEqual(r["over_2p5"], 0)

    def test_specials_and_signed_zero(self):
        for bits in (0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00000):
            self.assertEqual(self.measure(bits, bits)["special_mismatch"], 0)
        self.assertEqual(self.measure(0x80000000, 0)["special_mismatch"], 1)


if __name__ == "__main__":
    unittest.main()

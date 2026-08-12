from __future__ import annotations

import unittest

from flask_web.multi_gpu_server.consistency_check import (
    CompatibilityMismatch,
    compare_values,
)


class CompareValuesTest(unittest.TestCase):
    def test_accepts_small_float_difference_and_dynamic_timing(self) -> None:
        old = {"prediction": 0.900001, "processing_ms": 100, "label": 1}
        new = {"prediction": 0.900002, "processing_ms": 5, "label": 1}
        compare_values(old, new, tolerance=1e-5)

    def test_rejects_changed_keys_or_labels(self) -> None:
        with self.assertRaises(CompatibilityMismatch):
            compare_values({"label": 1}, {"label": 2})
        with self.assertRaises(CompatibilityMismatch):
            compare_values({"label": 1}, {"label": 1, "desc": "人类"})


if __name__ == "__main__":
    unittest.main()

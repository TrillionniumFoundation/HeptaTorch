"""Bounded input-control regressions without a native launch."""
import unittest
from unittest.mock import patch

import adaptive_pool as pool


class ScopeTests(unittest.TestCase):
    def test_tiling(self):
        self.assertEqual(pool.lowering([2, 3, 6, 8], [3, 2]), (2, 4))
        self.assertEqual(pool.lowering([1, 1, 7, 9], [1, 1]), (7, 9))
        self.assertEqual(pool.lowering([1, 1, 7, 9], [7, 9]), (1, 1))

    def test_unsupported_geometry_never_launches(self):
        cases = (
            ((1, 2, 5, 7), (2, 3)),
            ((1, 2, 2, 2), (3, 1)),
            ((1, 2, 2, 2), (2, 0)),
            ((1, 2, 2, 2), (True, 1)),
            ((2, 3, 4), (1, 1)),
            ((0, 1, 4, 4), (1, 1)),
            ((1, 1, 65, 1), (1, 1)),
            ((2, 1, 64, 64), (1, 1)),
        )
        with patch.object(pool, "pinned_bytes") as load:
            for shape, out in cases:
                with self.subTest(shape=shape, out=out), self.assertRaises(ValueError):
                    pool.adaptive_avg_pool2d("not accessed", shape, b"", out)
            load.assert_not_called()

    def test_wrong_buffer_extent_never_launches(self):
        with patch.object(pool, "pinned_bytes") as load:
            with self.assertRaises(ValueError):
                pool.adaptive_avg_pool2d("not accessed", (1, 1, 2, 2), b"", (1, 1))
            with self.assertRaises(ValueError):
                pool.adaptive_avg_pool2d("not accessed", (1, 1, 2, 2), bytes(16), (1, 1), seed=b"")
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()

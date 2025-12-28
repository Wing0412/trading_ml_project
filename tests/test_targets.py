import os
import sys

# Allow imports from project root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

import unittest
import pandas as pd
import numpy as np

from targets import add_future_returns_target


class TestTargets(unittest.TestCase):

    def setUp(self):
        n = 20
        dates = pd.date_range("2023-01-01", periods=n, freq="D")

        # Monotonic increasing prices (easy to reason about)
        close = np.arange(100, 100 + n)

        self.df = pd.DataFrame({
            "close": close
        }, index=dates)

    def test_target_columns_exist(self):
        out = add_future_returns_target(self.df, horizon=5)

        self.assertIn("future_return", out.columns)
        self.assertIn("target_direction", out.columns)

    def test_no_nans(self):
        out = add_future_returns_target(self.df, horizon=5)
        self.assertFalse(out.isna().any().any())

    def test_direction_is_positive_for_uptrend(self):
        out = add_future_returns_target(self.df, horizon=5)

        # Prices strictly increase → future_return must be positive
        self.assertTrue((out["target_direction"] == 1).all())

    def test_last_rows_removed(self):
        out = add_future_returns_target(self.df, horizon=5)

        # Last 5 rows should be removed
        self.assertEqual(len(out), len(self.df) - 5)

    def test_missing_close_raises_error(self):
        bad_df = pd.DataFrame({"price": [1, 2, 3]})

        with self.assertRaises(ValueError):
            add_future_returns_target(bad_df, horizon=5)


if __name__ == "__main__":
    unittest.main()

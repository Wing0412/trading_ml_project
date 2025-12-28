import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

import unittest
import pandas as pd
import numpy as np

from backtest import run_backtest


class TestBacktest(unittest.TestCase):

    def setUp(self):
        # Create a simple increasing price series
        n = 10
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        close = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109], index=dates)

        self.df = pd.DataFrame({"close": close}, index=dates)

    def test_backtest_runs_and_returns_metrics(self):
        # Fake regression prediction: always positive -> always long
        self.df["predicted_return"] = 0.01

        result = run_backtest(
            self.df,
            prediction_col="predicted_return",
            mode="regression",
            threshold=0.0
        )

        # Check outputs exist
        self.assertIsInstance(result.metrics, dict)
        self.assertIn("cumulative_return", result.metrics)
        self.assertIn("sharpe", result.metrics)

        # Equity curve length should match df length
        self.assertEqual(len(result.equity_curve), len(self.df))

    def test_shift_logic_no_lookahead(self):
        # Make positions long only starting from day 3 onward
        # If shift is correct, the first day with non-zero strategy return
        # occurs AFTER the first day with long signal.
        self.df["predicted_return"] = 0.0
        self.df.loc[self.df.index[2]:, "predicted_return"] = 0.01  # positive from day 3

        result = run_backtest(
            self.df,
            prediction_col="predicted_return",
            mode="regression",
            threshold=0.0
        )

        strat_ret = result.strategy_returns

        # Day 1 has NaN return; day 2 return should be 0 because position at day1 was 0
        # The first non-zero returns should occur starting day 4 (because of shift)
        # Explanation:
        # - Day 3 prediction => position=1 on Day 3
        # - position.shift(1) applies on Day 4 return
        nonzero_days = strat_ret.fillna(0).ne(0)
        if nonzero_days.any():
            first_nonzero_idx = np.where(nonzero_days.values)[0][0]
            self.assertGreaterEqual(first_nonzero_idx, 3)  # 0-based index: >=3 means Day 4+

    def test_classification_mode_probability(self):
        # Add a fake probability column
        self.df["predicted_direction"] = 1
        self.df["bullish_probability"] = 0.6  # above threshold

        result = run_backtest(
            self.df,
            prediction_col="predicted_direction",
            mode="classification",
            prob_col="bullish_probability",
            prob_threshold=0.55
        )

        # Should take long positions most days (after first day)
        self.assertTrue(result.positions.sum() > 0)


if __name__ == "__main__":
    unittest.main()

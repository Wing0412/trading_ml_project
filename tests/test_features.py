import os
import sys

# Make sure Python can find features.py in the project root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

import unittest
import numpy as np
import pandas as pd

from features import add_technical_features



class TestTechnicalFeatures(unittest.TestCase):

    def setUp(self):
        """
        Create a synthetic OHLCV DataFrame for indicator testing.
        """
        n = 400
        dates = pd.date_range(start="2020-01-01", periods=n, freq="D")

        close = np.linspace(100, 150, n) + np.random.normal(0, 1, n)
        high = close + np.random.uniform(0, 2, n)
        low = close - np.random.uniform(0, 2, n)
        open_ = close + np.random.normal(0, 0.5, n)
        volume = np.random.randint(100000, 500000, n)

        self.df = pd.DataFrame({
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume
        }, index=dates)

    def test_feature_columns_exist(self):
        out = add_technical_features(self.df.copy())

        expected_cols = [
            "ma_20", "ma_200", "ma_300",
            "bb_high", "bb_low",
            "rsi_14", "rsi_signal",
            "macd_line", "macd_hist",
            "vwap", "vol_change_pct",
            "atr_14",
            "pattern_hammer", "pattern_hanging_man",
            "pattern_engulfing", "pattern_shooting_star", "pattern_doji",
            "pattern_morning_star", "pattern_evening_star",
            "pattern_marubozu",
            "pattern_3_white_soldiers", "pattern_3_black_crows"
        ]

        for col in expected_cols:
            self.assertIn(col, out.columns, f"Missing feature: {col}")

    def test_no_nans_in_features(self):
        out = add_technical_features(self.df.copy())

        # After dropna(), no NaNs should remain
        self.assertFalse(out.isna().any().any())

    def test_empty_dataframe(self):
        empty = pd.DataFrame()
        result = add_technical_features(empty)
        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()

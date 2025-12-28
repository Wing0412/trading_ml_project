import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

import unittest
import numpy as np
import pandas as pd

from models import (
    train_direction_model,
    predict_direction_proba,
    train_return_model,
    predict_return
)


class TestModels(unittest.TestCase):

    def setUp(self):
        # Create a synthetic dataset with numeric "features" and targets.
        n = 300
        dates = pd.date_range("2020-01-01", periods=n, freq="D")

        # Synthetic features (numeric)
        x1 = np.linspace(0, 10, n)
        x2 = np.sin(np.linspace(0, 10, n))
        x3 = np.random.normal(0, 1, n)

        df = pd.DataFrame({
            "feat_x1": x1,
            "feat_x2": x2,
            "feat_x3": x3,
        }, index=dates)

        # Synthetic regression target (future_return)
        # Example: a simple function of features + noise
        df["future_return"] = 0.01 * df["feat_x1"] + 0.05 * df["feat_x2"] + 0.01 * df["feat_x3"]

        # Synthetic classification target (target_direction)
        # +1 if return > 0, else -1
        df["target_direction"] = np.where(df["future_return"] > 0, 1, -1)

        self.df = df

    def test_train_direction_model_returns_result(self):
        result = train_direction_model(self.df, test_size=0.2, model_kind="logreg")

        self.assertEqual(result.task, "classification")
        self.assertIn("accuracy", result.metrics)
        self.assertGreaterEqual(result.metrics["accuracy"], 0.0)
        self.assertLessEqual(result.metrics["accuracy"], 1.0)
        self.assertTrue(len(result.feature_columns) > 0)

    def test_predict_direction_proba_output(self):
        result = train_direction_model(self.df, test_size=0.2, model_kind="logreg")
        latest = self.df.tail(1)

        pred = predict_direction_proba(result, latest)

        self.assertIn("predicted_direction", pred)
        self.assertIn(pred["predicted_direction"], [-1.0, 1.0])

        # If model supports probability, it should be between 0 and 1
        if "bullish_probability" in pred:
            self.assertGreaterEqual(pred["bullish_probability"], 0.0)
            self.assertLessEqual(pred["bullish_probability"], 1.0)

    def test_train_return_model_returns_result(self):
        result = train_return_model(self.df, test_size=0.2, model_kind="ridge")

        self.assertEqual(result.task, "regression")
        self.assertIn("mae", result.metrics)
        self.assertIn("rmse", result.metrics)
        self.assertIn("r2", result.metrics)
        self.assertTrue(len(result.feature_columns) > 0)

    def test_predict_return_output(self):
        result = train_return_model(self.df, test_size=0.2, model_kind="ridge")
        latest = self.df.tail(1)

        pred = predict_return(result, latest)

        self.assertIsInstance(pred, float)


if __name__ == "__main__":
    unittest.main()

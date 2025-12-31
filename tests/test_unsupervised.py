import unittest
import numpy as np
import pandas as pd

from unsupervised import fit_kmeans_regimes


class TestUnsupervised(unittest.TestCase):
    def setUp(self):
        n = 400
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        # fake numeric features
        self.df = pd.DataFrame({
            "close": np.linspace(100, 150, n),
            "ma_20": np.linspace(100, 150, n) + np.random.normal(0, 0.1, n),
            "rsi_14": np.random.uniform(30, 70, n),
            "macd_hist": np.random.normal(0, 1, n),
            "volume": np.random.randint(1000, 2000, n),
        }, index=idx)

    def test_returns_cluster_column(self):
        out, res = fit_kmeans_regimes(self.df, k_values=[2, 3, 4])
        self.assertIn("cluster", out.columns)
        self.assertIn(res.best_k, [2, 3, 4])
        self.assertTrue("silhouette" in res.metrics)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            fit_kmeans_regimes(pd.DataFrame())


if __name__ == "__main__":
    unittest.main()

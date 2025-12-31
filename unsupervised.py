from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class UnsupervisedResult:
    method: str               # e.g. "kmeans"
    best_k: int
    metrics: Dict[str, float] # e.g. {"silhouette": 0.42}
    model: object             # fitted sklearn model (Pipeline)
    feature_columns: List[str]


def _select_feature_columns_unsupervised(df: pd.DataFrame) -> List[str]:
    """
    Select numeric feature columns and exclude obvious non-features / targets.
    """
    exclude = {
        "future_return",
        "target_direction",
        "predicted_direction",
        "bullish_probability",
        "cluster",  # if re-running
    }
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in exclude]


def fit_kmeans_regimes(
    df: pd.DataFrame,
    k_values: List[int] = [2, 3, 4, 5, 6],
    random_state: int = 42,
) -> Tuple[pd.DataFrame, UnsupervisedResult]:
    """
    Unsupervised learning: clusters days into 'regimes' using KMeans on features.

    Returns:
      - df_out: original df plus a 'cluster' column (0..k-1)
      - result: contains best_k, silhouette score, and fitted model
    """
    if df is None or df.empty:
        raise ValueError("Empty DataFrame passed to unsupervised learning.")

    feature_cols = _select_feature_columns_unsupervised(df)
    if not feature_cols:
        raise ValueError("No numeric feature columns found for clustering.")

    # Drop rows that still contain NaNs in any feature column
    work = df.copy()
    work = work.dropna(subset=feature_cols)
    if work.empty:
        raise ValueError("After dropping NaNs, no rows remain for clustering.")

    X = work[feature_cols].values

    best_k = None
    best_score = -1.0
    best_model = None

    # Try multiple k values and choose the best silhouette score
    for k in k_values:
        if k <= 1 or k >= len(work):
            continue

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=k, random_state=random_state, n_init=10))
        ])

        labels = model.fit_predict(X)

        # Silhouette requires at least 2 clusters and not all points in one cluster
        if len(set(labels)) < 2:
            continue

        score = silhouette_score(model.named_steps["scaler"].transform(X), labels)

        if score > best_score:
            best_score = float(score)
            best_k = int(k)
            best_model = model

    if best_model is None or best_k is None:
        raise ValueError("Could not fit KMeans with the provided k_values.")

    # Fit best model and attach labels to output df
    final_labels = best_model.fit_predict(X)
    work_out = work.copy()
    work_out["cluster"] = final_labels.astype(int)

    result = UnsupervisedResult(
        method="kmeans",
        best_k=best_k,
        metrics={"silhouette": best_score},
        model=best_model,
        feature_columns=feature_cols,
    )

    return work_out, result

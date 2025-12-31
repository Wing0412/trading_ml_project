from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# ----------------------------
# Config / Outputs
# ----------------------------

@dataclass
class ModelResult:
    task: str  # "classification" or "regression"
    model_name: str
    metrics: Dict[str, float]
    model: object
    feature_columns: List[str]


def _select_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Picks numeric feature columns and excludes obvious non-features / target columns.
    Adjust exclusions if you add more target columns later.
    """
    exclude = {
        "future_return",
        "target_direction",
        "target",  # if you later rename
    }
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude]
    return feature_cols


def _time_series_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple chronological split: last test_size portion is test.
    """
    if df is None or df.empty:
        raise ValueError("Empty DataFrame passed to model training.")

    n = len(df)
    split_idx = int(n * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df



# Classification: Bullish/Bearish


def train_direction_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    model_kind: str = "logreg",
) -> ModelResult:
    """
    Predicts target_direction: +1 bullish, -1 bearish (or 0 if you later add neutral).
    Returns a fitted model + test metrics.
    """
    if "target_direction" not in df.columns:
        raise ValueError("DataFrame missing 'target_direction'. Run targets.py first.")

    feature_cols = _select_feature_columns(df)
    if not feature_cols:
        raise ValueError("No numeric feature columns found to train on.")

    train_df, test_df = _time_series_train_test_split(df, test_size=test_size)

    X_train = train_df[feature_cols].values
    y_train = train_df["target_direction"].values

    X_test = test_df[feature_cols].values
    y_test = test_df["target_direction"].values

    if model_kind == "logreg":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000))
        ])
        model_name = "LogisticRegression"
    elif model_kind == "rf":
        model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            max_depth=6,
            min_samples_leaf=20,
            n_jobs=-1,
        )
        model_name = "RandomForestClassifier"
    else:
        raise ValueError("model_kind must be 'logreg' or 'rf'.")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    metrics = {
        "accuracy": float(acc),
    }

    # Optional: add more detail if you want it printed in main.py later
    # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred))

    return ModelResult(
        task="classification",
        model_name=model_name,
        metrics=metrics,
        model=model,
        feature_columns=feature_cols,
    )


def predict_direction_proba(
    model_result: ModelResult,
    df_latest: pd.DataFrame,
) -> Dict[str, float]:
    """
    Predict bullish/bearish on the most recent row(s).
    Returns:
      - predicted_direction (+1 or -1)
      - bullish_probability (if supported)
    """
    model = model_result.model
    cols = model_result.feature_columns

    X = df_latest[cols].values

    pred = model.predict(X)[-1]
    out = {"predicted_direction": float(pred)}

    # Some models support predict_proba (LogReg, RF classifier)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[-1]
        # Map probability to class +1 if possible
        classes = getattr(model, "classes_", None)
        if classes is not None and 1 in list(classes):
            idx = list(classes).index(1)
            out["bullish_probability"] = float(proba[idx])
        else:
            # fallback: just return max probability
            out["bullish_probability"] = float(np.max(proba))

    return out



# Regression: 1-week future_return


def train_return_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    model_kind: str = "ridge",
) -> ModelResult:
    """
    Predicts future_return (e.g., 5 trading days ahead return).
    Returns a fitted model + regression metrics.
    """
    if "future_return" not in df.columns:
        raise ValueError("DataFrame missing 'future_return'. Run targets.py first.")

    feature_cols = _select_feature_columns(df)
    if not feature_cols:
        raise ValueError("No numeric feature columns found to train on.")

    train_df, test_df = _time_series_train_test_split(df, test_size=test_size)

    X_train = train_df[feature_cols].values
    y_train = train_df["future_return"].values.astype(float)

    X_test = test_df[feature_cols].values
    y_test = test_df["future_return"].values.astype(float)

    if model_kind == "ridge":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0))
        ])
        model_name = "Ridge"
    elif model_kind == "rf":
        model = RandomForestRegressor(
            n_estimators=400,
            random_state=42,
            max_depth=8,
            min_samples_leaf=10,
            n_jobs=-1,
        )
        model_name = "RandomForestRegressor"
    else:
        raise ValueError("model_kind must be 'ridge' or 'rf'.")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }

    return ModelResult(
        task="regression",
        model_name=model_name,
        metrics=metrics,
        model=model,
        feature_columns=feature_cols,
    )


def predict_return(
    model_result: ModelResult,
    df_latest: pd.DataFrame,
) -> float:
    """
    Predicts future_return for the most recent row.
    """
    model = model_result.model
    cols = model_result.feature_columns
    X = df_latest[cols].values
    pred = model.predict(X)[-1]
    return float(pred)

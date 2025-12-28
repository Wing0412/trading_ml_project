from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    strategy_returns: pd.Series
    positions: pd.Series
    metrics: Dict[str, float]


def _compute_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Max drawdown in decimal form (e.g., -0.25 means -25% peak-to-trough).
    """
    running_max = equity_curve.cummax()
    drawdown = (equity_curve / running_max) - 1.0
    return float(drawdown.min())


def _compute_metrics(strategy_returns: pd.Series, periods_per_year: int = 252) -> Dict[str, float]:
    """
    Computes basic performance metrics from daily strategy returns.
    """
    strategy_returns = strategy_returns.dropna()

    if strategy_returns.empty:
        return {
            "cumulative_return": 0.0,
            "cagr": 0.0,
            "volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
        }

    equity = (1.0 + strategy_returns).cumprod()

    cumulative_return = float(equity.iloc[-1] - 1.0)

    # CAGR
    years = len(strategy_returns) / periods_per_year
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else 0.0

    # Volatility (annualized)
    vol = float(strategy_returns.std(ddof=0) * np.sqrt(periods_per_year))

    # Sharpe (risk-free assumed 0)
    sharpe = float((strategy_returns.mean() / strategy_returns.std(ddof=0)) * np.sqrt(periods_per_year)) \
        if strategy_returns.std(ddof=0) > 0 else 0.0

    max_dd = _compute_max_drawdown(equity)

    return {
        "cumulative_return": cumulative_return,
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }


def generate_positions_from_predictions(
    df: pd.DataFrame,
    prediction_col: str,
    mode: str = "regression",
    threshold: float = 0.0,
    prob_col: Optional[str] = None,
    prob_threshold: float = 0.55,
) -> pd.Series:
    """
    Converts model outputs to positions.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain prediction columns and 'close' for alignment.
    prediction_col : str
        Column name containing either predicted return (regression)
        or predicted direction (+1/-1) (classification).
    mode : str
        "regression" or "classification".
    threshold : float
        For regression: go long if predicted_return > threshold, else flat.
    prob_col : Optional[str]
        For classification: if provided, use probability threshold instead of raw direction.
    prob_threshold : float
        For classification with prob_col: go long if P(bullish) >= prob_threshold else flat.

    Returns
    -------
    pd.Series of positions: +1 for long, 0 for flat
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)

    if prediction_col not in df.columns and (prob_col is None or prob_col not in df.columns):
        raise ValueError("Prediction column(s) not found in DataFrame.")

    if mode not in {"regression", "classification"}:
        raise ValueError("mode must be 'regression' or 'classification'.")

    pos = pd.Series(0, index=df.index, dtype=float)

    if mode == "regression":
        # Long when predicted return is positive (or above threshold)
        pos.loc[df[prediction_col] > threshold] = 1.0

    else:
        # Classification mode
        if prob_col is not None:
            pos.loc[df[prob_col] >= prob_threshold] = 1.0
        else:
            # Long when predicted direction is bullish (+1)
            pos.loc[df[prediction_col] > 0] = 1.0

    return pos


def run_backtest(
    df: pd.DataFrame,
    prediction_col: str,
    mode: str = "regression",
    threshold: float = 0.0,
    prob_col: Optional[str] = None,
    prob_threshold: float = 0.55,
    initial_capital: float = 1.0,
) -> BacktestResult:
    """
    Runs a simple daily backtest:
    - Create positions from predictions
    - Apply positions (shifted by 1) to next-day close-to-close returns
    - Compute equity curve and metrics

    Notes
    -----
    Using position.shift(1) is critical: it prevents look-ahead.
    Today's prediction decides tomorrow's return exposure.
    """
    if df is None or df.empty:
        raise ValueError("Empty DataFrame passed to backtest.")

    if "close" not in df.columns:
        raise ValueError("DataFrame must contain 'close' for returns.")

    df = df.copy()

    # 1) Compute close-to-close daily returns
    df["asset_return"] = df["close"].pct_change()

    # 2) Convert predictions to positions (+1 long, 0 flat)
    df["position"] = generate_positions_from_predictions(
        df=df,
        prediction_col=prediction_col,
        mode=mode,
        threshold=threshold,
        prob_col=prob_col,
        prob_threshold=prob_threshold,
    )

    # 3) Strategy returns: position at t applies to return from t -> t+1
    df["strategy_return"] = df["position"].shift(1) * df["asset_return"]

    # 4) Equity curve
    equity_curve = (1.0 + df["strategy_return"].fillna(0.0)).cumprod() * initial_capital

    # 5) Metrics
    metrics = _compute_metrics(df["strategy_return"])

    return BacktestResult(
        equity_curve=equity_curve,
        strategy_returns=df["strategy_return"],
        positions=df["position"],
        metrics=metrics,
    )

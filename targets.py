import pandas as pd


def add_future_returns_target(
    df: pd.DataFrame,
    horizon: int = 5
) -> pd.DataFrame:

    # Safety check
    if df is None or df.empty:
        return df

    if "close" not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column")

    df = df.copy()

    # --- 1. Future price ---
    df["future_close"] = df["close"].shift(-horizon)

    # --- 2. Future return (regression target) ---
    df["future_return"] = (df["future_close"] / df["close"]) - 1.0

    # --- 3. Directional classification target ---
    df["target_direction"] = 0
    df.loc[df["future_return"] > 0, "target_direction"] = 1
    df.loc[df["future_return"] < 0, "target_direction"] = -1

    # --- 4. Cleanup ---
    # Rows near the end cannot know the future
    df = df.dropna()

    return df


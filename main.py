import pandas as pd

from data_loader import load_price_data
from features import add_technical_features
from targets import add_future_returns_target
from models import train_direction_model
from backtest import run_backtest
from datetime import date

def main():
    

    ticker = "SPY"                 
    start_date = "2015-01-01"   
    end_date = date.today().isoformat()  

    horizon = 5                    
    test_size = 0.2               
    model_kind = "logreg"          

    prob_threshold = 0.55         
    

    print(f"INFO: Loading data for {ticker}...")
    df = load_price_data(ticker, start_date, end_date)

    if df is None or df.empty:
        raise ValueError("FATAL: No price data loaded. Check ticker/date range/internet.")

    print("INFO: Adding technical features...")
    df = add_technical_features(df)

    print("INFO: Adding targets...")
    df = add_future_returns_target(df, horizon=horizon)

    # -----------------------------
    # TRAIN CLASSIFICATION MODEL
    # -----------------------------
    print("INFO: Training direction model...")
    model_result = train_direction_model(df, test_size=test_size, model_kind=model_kind)
    print(f"INFO: Model = {model_result.model_name}")
    print(f"INFO: Test metrics = {model_result.metrics}")

    # -----------------------------
    # GENERATE DAILY PROBABILITIES
    # -----------------------------
    # We predict probabilities on ALL rows (train+test) purely for backtest simulation.
    # For a serious research setup, you would do walk-forward (rolling retrain),
    # but this is the correct simple baseline.
    cols = model_result.feature_columns
    X_all = df[cols].values

    model = model_result.model

    if not hasattr(model, "predict_proba"):
        raise ValueError("This classification model does not support predict_proba(). Use logreg or rf.")

    proba = model.predict_proba(X_all)

    # Find which column corresponds to class +1 (bullish)
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Model classes do not include +1. Check your target_direction values.")

    bullish_idx = classes.index(1)

    df["bullish_probability"] = proba[:, bullish_idx]
    df["predicted_direction"] = model.predict(X_all)

    # -----------------------------
    # RUN BACKTEST
    # -----------------------------
    print("INFO: Running backtest...")
    bt = run_backtest(
        df=df,
        prediction_col="predicted_direction",   # required but not used if prob_col is set
        mode="classification",
        prob_col="bullish_probability",
        prob_threshold=prob_threshold,
        initial_capital=1.0
    )

    print("\n===== BACKTEST METRICS =====")
    for k, v in bt.metrics.items():
        print(f"{k}: {v}")

    # Optional: show last few rows to confirm signals
    print("\n===== SAMPLE OUTPUT (last 5 rows) =====")
    print(df[["close", "bullish_probability", "predicted_direction"]].tail())


if __name__ == "__main__":
    main()


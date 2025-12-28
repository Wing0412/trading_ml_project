import pandas as pd
import yfinance as yf
from typing import Dict, List
import os

# Define the folder where cached data will be stored
DATA_DIR = 'data/' 

def load_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Loads OHLCV price data for a single ticker, prioritizing cached CSV.
    Performs essential data cleaning and standardization.
    """
    # 1. Define filename for caching
    filename = os.path.join(DATA_DIR, f"{ticker}_data.csv")
    os.makedirs(DATA_DIR, exist_ok=True)

    # 2. Caching and download logic
    try:
        # Attempt to load from cache
        df = pd.read_csv(filename, parse_dates=True, index_col=0)
        print(f"INFO: Loaded {ticker} data from cache.")
    except FileNotFoundError:
        # Download data if cache doesn't exist
        print(f"INFO: Downloading {ticker} data from Yahoo Finance...")
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=False,   # keep Adj Close
                progress=False
            )
        except Exception as e:
            print(f"ERROR: Failed to download {ticker}. Reason: {e}")
            return pd.DataFrame()

        # Guardrail: ensure df is valid and not empty
        if not isinstance(df, pd.DataFrame) or df.empty:
            print(f"WARNING: Downloaded data for {ticker} is empty or invalid.")
            return pd.DataFrame()

        # Save to cache
        df.to_csv(filename, index=True)

    # 3. Column standardisation (handles MultiIndex and normal Index)
    if isinstance(df, pd.DataFrame):
        if isinstance(df.columns, pd.MultiIndex):
            # e.g. ('Adj Close','SPY') -> 'adj close'
            df.columns = [str(col[0]).lower() for col in df.columns]
        else:
            # e.g. 'Adj Close' -> 'adj close'
            df.columns = [str(col).lower() for col in df.columns]

    # 4. Fill missing values (forward fill), then drop any remaining NaNs
    df = df.ffill()
    df = df.dropna()

    # 5. Ensure sorted by date
    return df.sort_index()


def load_multi_asset_data(tickers: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """
    Loads price data for multiple assets and returns a dictionary of DataFrames.
    """
    all_data = {}
    for ticker in tickers:
        df = load_price_data(ticker, start, end)
        if not df.empty:
            all_data[ticker] = df
            
    # Check if any data was loaded successfully
    if not all_data:
        raise ValueError("FATAL ERROR: Could not load data for any specified ticker.")
        
    return all_data
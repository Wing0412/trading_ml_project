import pandas as pd

def load_price_data(path: str) -> pd.DataFrame:
    """
    Load OHLCV data from CSV.
    Expected columns: Date, Open, High, Low, Close, Volume
    """
    df = pd.read_csv(path, parse_dates=['Date'])
    df = df.sort_values('Date').set_index('Date')
    df = df.rename(columns=str.lower)
    return df

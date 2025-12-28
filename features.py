import pandas as pd
import ta
import talib  # <-- The library you successfully installed
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.volume import VolumeWeightedAveragePrice

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates and adds common technical indicators and candlestick patterns to 
    the price DataFrame.
    """
    
    # Safety Check
    if df is None or df.empty:
        return df

    # --- 1. Custom Moving Averages (Trend Indicators) ---
    df['ma_20'] = df['close'].rolling(window=20).mean()
    df['ma_200'] = df['close'].rolling(window=200).mean()
    df['ma_300'] = df['close'].rolling(window=300).mean() # Very long-term trend filter
    
    # --- 2. Bollinger Bands (Volatility Indicator) ---
    bb_indicator = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_high'] = bb_indicator.bollinger_hband()
    df['bb_low'] = bb_indicator.bollinger_lband()

    # --- 3. Relative Strength Index (Momentum Indicator) ---
    df["rsi_14"] = ta.momentum.RSIIndicator(close=df["close"], window=14).rsi()

    
    # RSI Overbought/Oversold Signal (Custom Range)
    df['rsi_signal'] = 0
    df.loc[df['rsi_14'] >= 80, 'rsi_signal'] = 2  # Extreme Overbought
    df.loc[(df['rsi_14'] >= 70) & (df['rsi_14'] < 80), 'rsi_signal'] = 1 # Overbought
    df.loc[df['rsi_14'] <= 40, 'rsi_signal'] = -1 # Oversold
    
    # --- 4. MACD (Trend/Momentum) ---
    macd_indicator = MACD(close=df['close'], window_fast=12, window_slow=26, window_sign=9)
    df['macd_line'] = macd_indicator.macd()
    df['macd_hist'] = macd_indicator.macd_diff()

    # --- 5. Volume Features ---
    df['vwap'] = VolumeWeightedAveragePrice(
        high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=14
    ).volume_weighted_average_price()
    
    df['vol_change_pct'] = df['volume'].pct_change()
    
    # --- 6. Pattern-Related Features ---
    # ATR is a volatility proxy useful for detecting pattern consolidation/breakouts
    df['atr_14'] = ta.volatility.AverageTrueRange(
        high=df['high'], low=df['low'], close=df['close'], window=14, fillna=False
    ).average_true_range()

    # --- 7. Candlestick Pattern Recognition (using talib) ---
    # Define OHLC variables for cleaner talib function calls
    o = df['open']
    h = df['high']
    l = df['low']
    c = df['close']
    
    # Reversal and Indecision Patterns
    df['pattern_hammer'] = talib.CDLHAMMER(o, h, l, c)
    df['pattern_hanging_man'] = talib.CDLHANGINGMAN(o, h, l, c)
    df['pattern_engulfing'] = talib.CDLENGULFING(o, h, l, c)
    df['pattern_shooting_star'] = talib.CDLSHOOTINGSTAR(o, h, l, c)
    df['pattern_doji'] = talib.CDLDOJI(o, h, l, c)
    
    # Strong Reversal/Continuation Patterns
    df['pattern_morning_star'] = talib.CDLMORNINGSTAR(o, h, l, c)
    df['pattern_evening_star'] = talib.CDLEVENINGSTAR(o, h, l, c)
    df['pattern_marubozu'] = talib.CDLMARUBOZU(o, h, l, c)
    df['pattern_3_white_soldiers'] = talib.CDL3WHITESOLDIERS(o, h, l, c)
    df['pattern_3_black_crows'] = talib.CDL3BLACKCROWS(o, h, l, c)

    # --- 8. Final Cleanup ---
    # Drop all NaN values created by the rolling windows (especially MA_300)
    df = df.dropna()
    
    return df

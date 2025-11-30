import pandas as pd
import numpy as np

def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma50'] = df['close'].rolling(50).mean()
    df['ma200'] = df['close'].rolling(200).mean()

    df['ma10_ma20_diff'] = df['ma10'] - df['ma20']
    df['ma20_ma50_diff'] = df['ma20'] - df['ma50']
    df['price_above_200'] = (df['close'] > df['ma200']).astype(int)
    return df

def add_rsi(df: pd.DataFrame, period=14) -> pd.DataFrame:
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    df['vol_rel'] = df['volume'] / (df['vol_ma20'] + 1e-9)
    df['vol_spike'] = (df['vol_rel'] > 2).astype(int)
    return df

def detect_cup_handle_window(prices: pd.Series) -> int:
    if len(prices) < 40:
        return 0
    first = prices[:10]
    middle = prices[10:30]
    last = prices[30:]

    start_high = first.max()
    mid_low = middle.min()
    end_high = last.max()

    cond_highs = abs(start_high - end_high) / start_high < 0.05
    cond_low = mid_low < first.min() and mid_low < last.min()

    return int(cond_highs and cond_low)

def add_pattern_flags(df: pd.DataFrame) -> pd.DataFrame:
    df['cup_handle'] = 0
    lookback = 40

    for i in range(lookback, len(df)):
        window = df['close'].iloc[i - lookback : i]
        df.iloc[i, df.columns.get_loc('cup_handle')] = detect_cup_handle_window(window)

    df['desc_triangle'] = 0
    return df

def add_ma_stack_signals(df: pd.DataFrame) -> pd.DataFrame:
    cond_bull = (df['ma10'] > df['ma20']) & (df['ma20'] > df['ma50']) & (df['ma50'] > df['ma200'])
    cond_bear = (df['ma10'] < df['ma20']) & (df['ma20'] < df['ma50']) & (df['ma50'] < df['ma200'])

    df['ma_bull_stack'] = cond_bull.astype(int)
    df['ma_bear_stack'] = cond_bear.astype(int)

    df['ma10_cross_over_20'] = ((df['ma10'] > df['ma20']) &
                                (df['ma10'].shift(1) <= df['ma20'].shift(1))).astype(int)

    df['ma10_cross_under_20'] = ((df['ma10'] < df['ma20']) &
                                 (df['ma10'].shift(1) >= df['ma20'].shift(1))).astype(int)
    return df

def add_trend_pattern_flags(df: pd.DataFrame) -> pd.DataFrame:
    N = 20
    df['price_trend'] = df['close'].diff().rolling(N).sum()

    df['bullish_continuation'] = (df['price_trend'] > 0).astype(int)
    df['bearish_continuation'] = (df['price_trend'] < 0).astype(int)

    df['higher_low'] = df['low'] > df['low'].shift(5)
    df['bullish_reversal'] = ((df['price_trend'].shift(5) < 0) & df['higher_low']).astype(int)

    df['lower_high'] = df['high'] < df['high'].shift(5)
    df['bearish_reversal'] = ((df['price_trend'].shift(5) > 0) & df['lower_high']).astype(int)

    return df

def add_bullish_score(df: pd.DataFrame) -> pd.DataFrame:
    bull_raw = (
        2*df['ma_bull_stack'] +
        1.5*df['ma10_cross_over_20'] +
        1*df['rsi'].between(50,70).astype(int) +
        1*df['vol_spike'] +
        1.5*df['bullish_continuation'] +
        1.5*df['bullish_reversal'] +
        1*df.get('cup_handle',0)
    )

    bear_raw = (
        2*df['ma_bear_stack'] +
        1.5*df['ma10_cross_under_20'] +
        1*(df['rsi']<40).astype(int) +
        1*df['vol_spike'] +
        1.5*df['bearish_continuation'] +
        1.5*df['bearish_reversal'] +
        1*df.get('desc_triangle',0)
    )

    df['bullish_score_raw'] = bull_raw - bear_raw

    min_v = df['bullish_score_raw'].min()
    max_v = df['bullish_score_raw'].max()
    df['bullish_score'] = (df['bullish_score_raw'] - min_v) / (max_v - min_v + 1e-9)

    return df

def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_volume_features(df)
    df = add_ma_stack_signals(df)
    df = add_pattern_flags(df)
    df = add_trend_pattern_flags(df)
    df = add_bullish_score(df)
    return df

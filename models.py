from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

FEATURE_COLS = [
    # MA features
    'ma10','ma20','ma50','ma200',
    'ma10_ma20_diff','ma20_ma50_diff',
    'price_above_200',

    # Pattern / signal features
    'ma_bull_stack','ma_bear_stack',
    'ma10_cross_over_20','ma10_cross_under_20',

    # RSI + volume
    'rsi','vol_rel','vol_spike',

    # Cup-handle + trend patterns
    'cup_handle','desc_triangle',
    'bullish_continuation','bearish_continuation',
    'bullish_reversal','bearish_reversal',

    # Bullish score
    'bullish_score'
]

def make_regression_model():
    return RandomForestRegressor(n_estimators=300, random_state=42)

def make_classification_model():
    return RandomForestClassifier(n_estimators=300, random_state=42)

import pandas as pd
import numpy as np

def simulate_trades(df: pd.DataFrame,
                    y_reg_pred,
                    y_cls_proba,
                    horizon: int = 5,
                    cls_threshold: float = 0.6,
                    sl_pct: float = 0.05,
                    tp_pct: float = 0.08):
    df = df.copy()
    df = df.iloc[-len(y_reg_pred):]
    df['pred_return'] = y_reg_pred
    df['prob_up'] = y_cls_proba
    df['enter'] = (df['prob_up'] > cls_threshold).astype(int)

    equity = [1.0]
    position = 0
    closes = df['close'].values

    for i in range(len(df)):
        if position == 0 and df['enter'].iloc[i] == 1:
            entry_price = closes[i]
            entry_idx = i
            position = 1
        elif position == 1 and i >= entry_idx + horizon:
            exit_price = closes[i]
            ret = (exit_price - entry_price) / entry_price
            equity.append(equity[-1] * (1 + ret))
            position = 0

    df['equity'] = np.nan
    df.loc[df.index[:len(equity)], 'equity'] = equity
    return df

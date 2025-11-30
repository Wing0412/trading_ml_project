import numpy as np
import pandas as pd

def compute_forward_return(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    df['target_return_5d'] = np.log(df['close'].shift(-horizon) / df['close'])
    df['target_up_5d'] = (df['target_return_5d'] > 0).astype(int)
    return df

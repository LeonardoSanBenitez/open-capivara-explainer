import pandas as pd
import numpy as np


def pd_duplicate_rows(df: pd.DataFrame, duplication_key: str, n: int) -> pd.DataFrame:
    '''
    If the value of <duplication_key> is non-nan (anything that is filled, except empty strings), then to duplicate that row n times

    No side effects
    '''
    assert n > 1, "Multiplier should be at least 2"
    assert df.shape[0] > 0, "Empty DataFrame"
    mask = df[duplication_key].notna() & (df[duplication_key] != '')
    duplicated_df = df[mask].reindex(df[mask].index.repeat(n-1)).reset_index(drop=True)  # -1 because the original row is concatenated in the end
    result_df = pd.concat([df, duplicated_df], ignore_index=True)
    result_df = result_df.sample(frac=1).reset_index(drop=True)  # suffle
    return result_df

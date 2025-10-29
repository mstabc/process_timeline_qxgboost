import pandas as pd
from eventlog_pipeline.utils.helpers import _standard_cols

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input columns: ['case:concept:name','concept:name','time:timestamp'] (standardized in loader)
    
    drop NA, 
    drop duplicates,
    sort by (case_id, timestamp).
    """
    df = _standard_cols(df)
    
    # No NA in key columns
    df = df.dropna(subset=["case_id", "task", "timestamp"]).copy()

    # No duplicates
    df = df.drop_duplicates(subset=["case_id", "task", "timestamp"])

    # Sort (We need this because for start date we use shift in transform.py) (timestamp of each event (task) in a case is complete date of itself but it is the start date of next event)
    df = df.sort_values(["case_id", "timestamp"]).reset_index(drop=True)



    return df

import pandas as pd
from typing import Optional, Iterable
import os
from pathlib import Path


def drop_cols(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)


#We should standardize column names to 'case:concept:name', 'concept:name', 'time:timestamp' with a proper formatting and types
_CASE_CANDIDATES = ["case:concept:name","case_id","case","caseid","case id","case-number"]
_TASK_CANDIDATES = ["concept:name","activity","task","event","event_name","event name"]
_TIME_CANDIDATES = ["time:timestamp","timestamp","complete","end_time","end","time","date","event_time"]

def _pick_column(cols: Iterable[str], candidates: Iterable[str]) -> Optional[str]:

    # If exact match exists, return it
    m = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in m: return m[cand.lower()]

    # in case there is a messed up spacing or uppercase in the column names of the dataset
    for c in cols:
        cl = c.lower().replace(" ", "")
        for cand in candidates:
            if cand.lower().replace(" ", "") in cl:
                return c
    return None

def _standard_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Identify and standardize key event log columns.
    cols = list(df.columns)
    case_col = _pick_column(cols, _CASE_CANDIDATES)
    task_col = _pick_column(cols, _TASK_CANDIDATES)
    time_col = _pick_column(cols, _TIME_CANDIDATES)
    
    # The names we should have
    mapping = {
        case_col: "case:concept:name",
        task_col: "concept:name",
        time_col: "time:timestamp"
    }

    missing = [std for std, real in mapping.items() if std is None]
    if missing:
        raise KeyError(f"We need this column in the dataset: {missing}")
    
    df = pd.DataFrame({
        "case:concept:name": df[case_col].astype("string", copy=False),
        "concept:name": df[task_col].astype("string", copy=False),
        "time:timestamp": pd.to_datetime(df[time_col], errors="coerce", utc=True),
    })
    df = df.rename(columns={
        "case:concept:name": "case_id",
        "concept:name": "task",
        "time:timestamp": "timestamp",
    })
    
    return df

# iterrows() with repeated filtering is extremely slow (O(n²) complexity) for large dataframes.
def _task_before(df, time_key_func, col_name):
    counts = []
    for i, row in df.iterrows():
        task = row["task"]
        create_time = row["create_date"]

        bucket_key = time_key_func(create_time)

        mask = (
            (df["task"] == task)
            & (df["complete_date"] < create_time)
            & (df["complete_date"].apply(time_key_func) == bucket_key)
        )
        counts.append(mask.sum())

    df[col_name] = counts
    return df

def _case_before(df, time_key_func, col_name):
        """
        Compute number of unique cases completed before current row's create_date
        within the same time bucket.
        """
        #TODO: Optimize this at least to skip the pre computed case ids
        counts = []
        for i, row in df.iterrows():
            create_time = row["create_date"]
            bucket_key = time_key_func(create_time)

            mask = (
                (df["case_complete_date"] < create_time)
                & (df["case_complete_date"].apply(time_key_func) == bucket_key)
            )

            unique_case_count = df.loc[mask, "case_id"].nunique()
            counts.append(unique_case_count)
            
        df[col_name] = counts
        return df
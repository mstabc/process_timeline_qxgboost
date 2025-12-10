import pandas as pd
from typing import Optional, Iterable
import os
from pathlib import Path
import numpy as np

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


def task_before_fast(df, time_key_func, col_name):
    """
    Vectorized equivalent of _task_before.

    For each row in df, count how many *completed* rows exist with:
      - same task
      - same time bucket (defined by time_key_func on create/complete date)
      - complete_date < this row's create_date
    and store that count in df[col_name].

    Idea:
      Instead of looping over all rows and re-counting matches each time (O(n^2)),
      we precompute, per (task, bucket), a time-ordered cumulative count of
      completions (`comp_count`), then use `merge_asof` to look up the last
      completion before each create_date. This turns the problem into ~O(n log n).
    """
    df = df.copy()

    # Precompute bucket for create and complete dates separately
    df["bucket_create"] = df["create_date"].map(time_key_func)
    df["bucket_complete"] = df["complete_date"].map(time_key_func)

    # Output series (same index as df)
    out_counts = pd.Series(0, index=df.index, dtype="int64")

    # Process each task separately to avoid multi-key merge_asof headaches
    for task, df_task in df.groupby("task", sort=False):

        # --- completion side for this task ---
        comp = df_task.loc[df_task["complete_date"].notna(),
                           ["complete_date", "bucket_complete"]].copy()
        if comp.empty:
            # no completed rows for this task -> all zeros for this task
            continue

        comp = comp.rename(columns={"bucket_complete": "bucket"})
        # sort within each bucket by complete_date
        comp = comp.sort_values(["bucket", "complete_date"])

        # cumulative count of completions per (task is fixed here, so per bucket)
        comp["comp_count"] = comp.groupby("bucket").cumcount() + 1

        # merge_asof with by="bucket" requires sort by [bucket, complete_date]
        comp = comp.sort_values(["bucket", "complete_date"])

        # --- create side for this task ---
        create = df_task[["create_date", "bucket_create"]].copy()
        create = create.rename(columns={"bucket_create": "bucket"})
        create["orig_index"] = create.index  # remember original positions

        create = create.sort_values(["bucket", "create_date"])

        # --- as-of join: for each (bucket, create_date) find last completion before it ---
        merged = pd.merge_asof(
            create,
            comp,
            left_on="create_date",
            right_on="complete_date",
            by="bucket",
            direction="backward",
            allow_exact_matches=False,   # strict "<", same as original mask
        )

        # comp_count = number of prior completions in that bucket for this task
        counts_task = (
            merged.set_index("orig_index")["comp_count"]
            .fillna(0)
            .astype("int64")
        )

        # write into overall output (indexes are original df indexes)
        out_counts.loc[counts_task.index] = counts_task

    df[col_name] = out_counts

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


def case_before_fast(df, time_key_func, col_name):
    """
    For each row in df, compute how many *distinct case_id* values
    have case_complete_date < create_date in the same time bucket
    (bucket defined by time_key_func).

    Idea:
      Instead of scanning all rows per create_date O(n^2), we first build a per-bucket,
      time-ordered list of case completions with a cumulative `case_count`, then use merge_asof to
      look up the last completion before each create_date. That turns it into ~O(n log n).
    """

    # --- 1) Build per-case completion table ---
    # Keep rows where we know when the case was completed.
    # Assume case_complete_date is the same for a given case_id.
    case_complete = df.loc[df["case_complete_date"].notna(),
                           ["case_id", "case_complete_date"]].copy()

    # One row per case_id, using earliest completion if duplicates exist.
    case_complete = (
        case_complete
        .sort_values("case_complete_date")
        .drop_duplicates(subset="case_id", keep="first")
    )

    # Assign bucket per case based on its completion time
    case_complete["bucket"] = case_complete["case_complete_date"].map(time_key_func)

    # --- 2) Within each bucket, sort by completion time and cumulative count cases ---
    case_complete = case_complete.sort_values(["bucket", "case_complete_date"])
    case_complete["case_count"] = (
        case_complete.groupby("bucket").cumcount() + 1  # 1-based cumulative #cases
    )

    # For merge_asof, right side must be sorted by the 'on' key
    case_complete = case_complete.sort_values("case_complete_date")

    # --- 3) Build left table: one row per create_date we want to evaluate ---
    left = df[["create_date"]].copy()
    left["bucket"] = left["create_date"].map(time_key_func)
    left["orig_index"] = left.index  # so we can restore original order

    # merge_asof requires left_on sorted too
    left = left.sort_values("create_date")

    # --- 4) As-of join: for each create_date, last case completed before it in that bucket ---
    merged = pd.merge_asof(
        left,
        case_complete,
        left_on="create_date",
        right_on="case_complete_date",
        by="bucket",
        direction="backward",          # strictly before (because of allow_exact_matches=False)
        allow_exact_matches=False,
    )

    # case_count at that matched row = #distinct cases completed before create_date in that bucket
    counts = (
        merged.set_index("orig_index")["case_count"]
        .fillna(0)
        .astype("int64")
        .reindex(df.index)  # back to original df row order
    )

    df[col_name] = counts
    return df


def build_duration_classes_quantiles(
    durations: pd.Series,
    n_bins: int = 5,
    zero_threshold: float = 0.0,
):
    """
    Build duration classes using quantiles on non-zero values.

    Returns:
      labels: pd.Series of int (0 for zero, 1..n_bins for non-zero)
      bin_edges_raw: np.ndarray of bin edges in original seconds
    """
    durations = pd.to_numeric(durations, errors="coerce").fillna(0)
    zero_mask = durations <= zero_threshold
    non_zero = durations[~zero_mask]

    if non_zero.empty:
        return pd.Series(0, index=durations.index, dtype=int), np.array([])

    # Work in log-space for smoother bins
    log_non_zero = np.log1p(non_zero)

    # Quantile edges in log-space
    q = np.linspace(0.0, 1.0, n_bins + 1)
    log_edges = np.quantile(log_non_zero, q)

    # Ensure strictly increasing edges to avoid cut() issues
    log_edges = np.unique(log_edges)
    if len(log_edges) - 1 < n_bins:
        # If not enough unique edges (e.g., many ties), reduce bin count
        n_bins = len(log_edges) - 1

    # Cut in log-space
    labels_non_zero = pd.cut(
        log_non_zero,
        bins=log_edges,
        labels=range(1, n_bins + 1),
        include_lowest=True,
    ).astype(int)

    labels = pd.Series(0, index=durations.index, dtype=int)
    labels.loc[~zero_mask] = labels_non_zero

    # For interpretability, map bin edges back to seconds
    bin_edges_raw = np.expm1(log_edges)

    return labels, bin_edges_raw


def assign_duration_class_from_edges(sec: float, edges: np.ndarray) -> int:
    if sec <= 0:
        return 0
    log_sec = np.log1p(sec)
    log_edges = np.log1p(edges)
    # bins: 1..n_bins
    bin_idx = np.digitize([log_sec], log_edges[1:-1], right=True)[0] + 1
    return int(bin_idx)
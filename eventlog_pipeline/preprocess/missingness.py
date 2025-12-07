import numpy as np
import pandas as pd


def introduce_missing_values(
    df: pd.DataFrame,
    target_cols,
    case_col: str = "case_id",
    time_col: str = "timestamp",   # or 'complete_date'
    missing_rate: float = 0.3,
    random_state: int = 42,
    inplace: bool = False,
):
    """
    Introduce MCAR missingness in one or more columns, avoiding the first
    and last time points per case (based on `time_col`).

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataset with at least [case_col, time_col] and the target columns.
    target_cols : str or list of str
        Column name or list of column names in which to introduce missing values.
    case_col : str, default 'case_id'
        Column identifying each case / sequence.
    time_col : str, default 'timestamp'
        Time column used to determine first/last time points per case.
    missing_rate : float, default 0.3
        Fraction of *eligible* values (excluding first/last per case and existing NaNs)
        to set to NaN in each target column (independently).
    random_state : int, default 42
        Random seed for reproducibility.
    inplace : bool, default False
        If True, modify `df` in place. If False, work on a copy.

    Returns
    -------
    df_out : pd.DataFrame
        DataFrame with induced missingness in `target_cols`.
    masks : dict[str, pd.Series]
        Dictionary mapping each target column to a boolean mask (index-aligned
        with df_out) indicating which entries were set to NaN by this function.
    """
    if isinstance(target_cols, str):
        target_cols = [target_cols]
    else:
        target_cols = list(target_cols)

    for col in target_cols:
        if col not in df.columns:
            raise ValueError(f"target column '{col}' not found in DataFrame")

    if case_col not in df.columns:
        raise ValueError(f"case_col '{case_col}' not found in DataFrame")

    if time_col not in df.columns:
        raise ValueError(f"time_col '{time_col}' not found in DataFrame")

    # Work on a copy unless explicitly told not to
    if not inplace:
        df = df.copy()

    # Sort by case + time, but keep original index
    df_sorted = df.sort_values([case_col, time_col]).reset_index(drop=False)
    original_index = df_sorted["index"]
    df_sorted = df_sorted.drop(columns=["index"])

    # First/last row per case (based on time_col)
    grouped = df_sorted.groupby(case_col)[time_col]
    first_idx = grouped.idxmin()
    last_idx = grouped.idxmax()
    protected_idx = pd.Index(first_idx.values).union(
        pd.Index(last_idx.values)
    )

    rng = np.random.RandomState(random_state)
    masks_local = {}

    for col in target_cols:
        # Eligible positions: not first/last in case, and not already NaN
        eligible_mask = (~df_sorted.index.isin(protected_idx)) & df_sorted[col].notna()
        eligible_idx = df_sorted.index[eligible_mask]

        n_candidates = len(eligible_idx)
        if n_candidates == 0:
            # nothing to mask in this column
            mask_col = pd.Series(False, index=df_sorted.index)
            masks_local[col] = mask_col
            continue

        n_missing = int(round(missing_rate * n_candidates))
        n_missing = max(0, min(n_missing, n_candidates))

        if n_missing == 0:
            mask_col = pd.Series(False, index=df_sorted.index)
            masks_local[col] = mask_col
            continue

        missing_idx = rng.choice(eligible_idx, size=n_missing, replace=False)

        # Apply masking
        df_sorted.loc[missing_idx, col] = np.nan

        mask_col = pd.Series(False, index=df_sorted.index)
        mask_col.loc[missing_idx] = True
        masks_local[col] = mask_col

    # Map back to original index order
    df_sorted.index = original_index
    df_out = df_sorted.sort_index()

    masks = {}
    for col, m in masks_local.items():
        m.index = original_index
        masks[col] = m.sort_index()

    return df_out, masks

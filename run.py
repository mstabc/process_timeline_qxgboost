import argparse
import joblib

import pandas as pd
import numpy as np

from pathlib import Path

from eventlog_pipeline.utils.logging import setup_logger
from eventlog_pipeline.io.loader import load_file
from eventlog_pipeline.preprocess.cleaning import clean_data
from eventlog_pipeline.preprocess.transform import compute_durations_and_sequences, create_case_summary, prior_completions
from eventlog_pipeline.preprocess.workload import compute_workload
from eventlog_pipeline.preprocess.merge import merger
from eventlog_pipeline.utils.helpers import build_duration_classes_quantiles
from eventlog_pipeline.preprocess.missingness import introduce_missing_values

from eventlog_pipeline.train.predictors import (
    build_sequence_predictors,
    build_duration_class_predictors,
    build_duration_regression_predictors,
)
from eventlog_pipeline.train.modeling import train_quantile_regressors, train_classifier_model
from eventlog_pipeline.train.plotting import (
    plot_prediction_intervals_candlestick,
    plot_interval_width_histogram,
    compute_quantile_interval_report,
    save_quantile_interval_report,
)

from sklearn.utils import shuffle


from eventlog_pipeline.preprocess.missingness import introduce_missing_values


def step_simulate_missing(
    merged_csv: Path,
    outputs_dir: Path,
    logger,
    target_col: str = "relative_start",
    missing_rate: float = 0.3,
    random_state: int = 42,
) -> Path:
    """
    Load the merged preprocessed dataset, introduce missing values in `target_col`
    (avoiding first/last time points per case_id based on `timestamp`),
    and save a new CSV + mask.

    """
    logger.info("Loading merged data for missingness simulation: %s", merged_csv)
    df = pd.read_csv(merged_csv)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in merged dataset")

    logger.info(
        "Introducing missingness in column '%s' with rate %.2f "
        "(excluding first/last event per case_id)",
        target_col, missing_rate
    )

    # Introduce missingness in the specified column
    df_miss, masks = introduce_missing_values(
        df,
        target_cols=target_col,         # can be str or list, helper handles both
        case_col="case_id",
        time_col="timestamp",           # or 'complete_date' if you prefer
        missing_rate=missing_rate,
        random_state=random_state,
    )

    pre_dir = outputs_dir / "preprocessed"
    pre_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(merged_csv).stem  # e.g. 'sepsis_merged'
    miss_suffix = f"miss_{target_col}_{int(missing_rate * 100)}"

    out_csv = pre_dir / f"{base_name}_{miss_suffix}.csv"
    df_miss.to_csv(out_csv, index=False)
    logger.info("Saved dataset with induced missingness → %s", out_csv)

    return out_csv

def step_preprocess(dataset: str, outputs_dir, logger):
    logger.info("Loading dataset '%s'...", dataset)
    
    raw = load_file(dataset)
    logger.info("Raw rows: %d", len(raw))
    
    # Pre processing steps
    pre_dir = outputs_dir / "preprocessed"

    df = clean_data(raw)
    df = compute_durations_and_sequences(df)
    df, in_throughput_cols = prior_completions(df)

    queue_df, in_queue_cols, progress_df, in_progress_cols = compute_workload(df)

    case_summary_df, exists_cols = create_case_summary(df)
    case_summary_df.to_csv(pre_dir / f"{dataset}_case_summary.csv", index=False)
    merged, label_encoder = merger(
        df, queue_df, progress_df, case_summary_df
    )
    
    joblib.dump(label_encoder, pre_dir / f"{dataset}_task_label_encoder.joblib")
    joblib.dump({"in_queue": in_queue_cols, "in_progress": in_progress_cols, "exists": exists_cols, "in_throughput": in_throughput_cols},
                pre_dir / f"{dataset}_column_groups.joblib")
    
    merged_path = pre_dir / f'{dataset}_merged.csv'
    merged.to_csv(merged_path, index=False)
    logger.info("Saved merged CSV %s", merged_path)
    return merged_path

def step_train_sequence(merged_csv, dataset, outputs_dir, logger):

    logger.info("Loading merged data: %s", merged_csv)
    merged_csv = Path(merged_csv)
    dataset = merged_csv.stem
    if dataset.endswith("_merged"):
        dataset = dataset[: -len("_merged")]
    merged = pd.read_csv(merged_csv)
    
    #Less than one year total duration cases
    merged = merged[merged['total_duration_sec'] < 31556926 ].copy()
    case_ids = shuffle(merged['case_id'].drop_duplicates(), random_state=42)
    num_total = len(case_ids)
    size_25pct = int(0.25 * num_total)

    case_ids_seq = case_ids.iloc[:size_25pct]
    case_ids_seq_pred = case_ids.iloc[size_25pct:]

    case_seq = merged[merged['case_id'].isin(case_ids_seq)].copy()
    case_seq_pred = merged[merged['case_id'].isin(case_ids_seq_pred)].copy()


    # Normalization
    for col in ["relative_start", "relative_complete"]:
        if col in case_seq.columns:
            case_seq[col] = pd.to_numeric(case_seq[col], errors="coerce").fillna(0)
            case_seq[col] = np.log1p(np.maximum(case_seq[col], 0))
        

    #Loading the col names
    col_groups = joblib.load(merged_csv.parent / f"{dataset}_column_groups.joblib")
    in_queue_cols    = [c for c in col_groups["in_queue"] if c in merged.columns]
    in_progress_cols = [c for c in col_groups["in_progress"] if c in merged.columns]
    exists_cols      = [c for c in col_groups["exists"] if c in merged.columns]
    in_throughput_cols = [c for c in col_groups["in_throughput"] if c in merged.columns]
    
    
    seq_predictors = build_sequence_predictors(in_progress_cols, in_queue_cols, exists_cols, in_throughput_cols)
    print(f"Sequence predictors: {seq_predictors}")


    models_dir = outputs_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    ### Here we train two separate models for start and complete sequences
    start_models, start_scaler, start_features, start_metrics = train_quantile_regressors(
        case_seq[[*seq_predictors,"relative_start"]].dropna(subset=["relative_start"]), dataset,
        seq_predictors, label="relative_start", out_dir=models_dir, quantiles=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    )
    
    comp_models, comp_scaler, comp_features, comp_metrics = train_quantile_regressors(
        case_seq[[*seq_predictors,"relative_complete"]].dropna(subset=["relative_complete"]), dataset,
        seq_predictors, label="relative_complete", out_dir=models_dir, quantiles=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    )

    #  Predict on the held-out set so that duration models can use these predictions
    X_pred = case_seq_pred[start_features]
    X_pred_scaled = start_scaler.transform(X_pred)
    case_seq_pred["predicted_start_sequence"] = start_models[0.5].predict(X_pred_scaled)

    X_pred2 = case_seq_pred[comp_features]
    X_pred2_scaled = comp_scaler.transform(X_pred2)
    case_seq_pred["predicted_complete_sequence"] = comp_models[0.5].predict(X_pred2_scaled)

    reports_dir = outputs_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {f"{dataset}_sequence_start": start_metrics, f"{dataset}_sequence_complete": comp_metrics},
        reports_dir / f"{dataset}_metrics_summary.joblib"
    )

    case_seq_pred.to_csv(outputs_dir / "preprocessed" / f"{dataset}_case_seq_pred.csv", index=False)
    logger.info("Saved case sequence predictions → %s", outputs_dir / "preprocessed" / f"{dataset}_case_seq_pred.csv")
def classify_binary_duration(v: float, zero_threshold: float = 1) -> int:
    """Zero vs non zero duration."""
    return 0 if v <= zero_threshold else 1

def step_train_duration(merged_csv, dataset, outputs_dir, logger):

    logger.info("Loading merged data for duration models: %s", merged_csv)
    merged = pd.read_csv(merged_csv)

    # ensure duration_sec exists as raw seconds
    if "duration_sec" not in merged.columns:
        merged["duration"] = pd.to_timedelta(merged["duration"])
        merged["duration_sec"] = merged["duration"].dt.total_seconds().fillna(0)

    # less than 6 months tasks
    merged = merged[merged["duration_sec"] < 1.577e7].copy()

    # split by unique case IDs
    case_ids = shuffle(merged["case_id"].drop_duplicates(), random_state=42)
    num_total = len(case_ids)
    size_25pct = int(0.25 * num_total)

    case_ids_bin_multi = case_ids.iloc[:size_25pct]
    case_ids_task_duration = case_ids.iloc[size_25pct:]

    df_bin_multi = merged[merged["case_id"].isin(case_ids_bin_multi)].copy()
    df_task_duration = merged[merged["case_id"].isin(case_ids_task_duration)].copy()

    # duration preprocessing: keep raw seconds and add log space for regression
    for df in [df_bin_multi, df_task_duration]:
        if "duration_sec" in df.columns:
            df["duration_sec"] = pd.to_numeric(df["duration_sec"], errors="coerce").fillna(0)
            df["log_duration_sec"] = np.log1p(np.maximum(df["duration_sec"], 0))

    col_groups = joblib.load(merged_csv.parent / f"{dataset}_column_groups.joblib")
    in_queue_cols      = [c for c in col_groups["in_queue"]      if c in merged.columns]
    in_progress_cols   = [c for c in col_groups["in_progress"]   if c in merged.columns]
    exists_cols        = [c for c in col_groups["exists"]        if c in merged.columns]
    in_throughput_cols = [c for c in col_groups["in_throughput"] if c in merged.columns]

    ducl_predictors = build_duration_class_predictors(
        in_progress_cols, in_queue_cols, exists_cols, in_throughput_cols
    )
    du_predictors = build_duration_regression_predictors(
        in_progress_cols, in_queue_cols, exists_cols, in_throughput_cols
    )

    models_dir = outputs_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # label generation using helpers (quantile based classes)
    logger.info("Building duration classes from quantiles")

    # binary: zero vs non zero on raw seconds
    df_bin_multi["binary_duration_class"] = df_bin_multi["duration_sec"].apply(
        classify_binary_duration
    )

    # multi class: quantiles on non zero raw durations
    multi_labels, bin_edges = build_duration_classes_quantiles(
        df_bin_multi["duration_sec"],
        n_bins=5,
        zero_threshold=0.0,
    )
    df_bin_multi["multi_duration_class"] = multi_labels

    logger.info("Duration class bin edges (seconds): %s", bin_edges)

    # persist bin edges for later inference with assign_duration_class_from_edges
    joblib.dump(
        {"bin_edges": bin_edges},
        models_dir / f"{dataset}_duration_bins.joblib",
    )

    print("Duration class distribution (binary):")
    print(df_bin_multi["binary_duration_class"].value_counts())
    print("Duration class distribution (multi):")
    print(df_bin_multi["multi_duration_class"].value_counts())

    # train classifiers
    logger.info("Training duration classifiers")

    if len(df_bin_multi["binary_duration_class"].unique()) == 2:
        bin_model, bin_scaler, bin_le = train_classifier_model(
            df_bin_multi,
            dataset,
            ducl_predictors,
            "binary_duration_class",
            out_dir=models_dir / "duration_binary",
        )
        X_all_bin = bin_scaler.transform(df_task_duration[ducl_predictors])
        df_task_duration["binary_pred"] = bin_le.inverse_transform(
            bin_model.predict(X_all_bin).astype(int)
        )
    else:
        logger.warning(
            "Skipping binary duration classifier training due to insufficient class variety."
        )
        df_task_duration["binary_pred"] = 1

    # train multi class only on non zero durations
    non_zero = df_bin_multi.query("binary_duration_class == 1")
    if non_zero["multi_duration_class"].nunique() > 1:
        multi_model, multi_scaler, multi_le = train_classifier_model(
            non_zero,
            dataset,
            ducl_predictors,
            "multi_duration_class",
            out_dir=models_dir / "duration_multi",
        )

        # classifier inference
        non_zero_mask = df_task_duration["binary_pred"] == 1
        X_all_multi = multi_scaler.transform(
            df_task_duration.loc[non_zero_mask, ducl_predictors]
        )
        df_task_duration.loc[non_zero_mask, "multi_pred"] = multi_le.inverse_transform(
            multi_model.predict(X_all_multi).astype(int)
        )
        df_task_duration.loc[~non_zero_mask, "multi_pred"] = 0
    else:
        logger.warning(
            "Skipping multi duration classifier training due to insufficient class variety."
        )
        df_task_duration["multi_pred"] = 0

    df_task_duration["predicted_duration_class"] = df_task_duration["multi_pred"].astype(int)

    # train duration regressors on log duration
    logger.info("Training task duration quantile regressor on log duration")
    task_models, task_scaler, task_features, task_metrics = train_quantile_regressors(
        df_task_duration[[*du_predictors, "log_duration_sec"]],
        dataset,
        du_predictors,
        label="log_duration_sec",
        out_dir=models_dir / "task_duration",
        quantiles=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    )

    # Quantile interval visuals + coverage report (based on duration predictions)
    X_all = task_scaler.transform(df_task_duration[task_features])
    pred_log = {q: model.predict(X_all) for q, model in task_models.items()}

    # Convert log-seconds to hours for readable plots and reports
    y_true_hours = pd.to_numeric(df_task_duration["duration_sec"], errors="coerce") / 3600.0
    pred_hours = {q: np.expm1(vals) / 3600.0 for q, vals in pred_log.items()}

    interval_dir = outputs_dir / "reports" / "quantile_intervals"
    interval_dir.mkdir(parents=True, exist_ok=True)

    n_samples = int(len(y_true_hours))
    max_points = min(300, max(50, int(n_samples * 0.05)))

    plot_prediction_intervals_candlestick(
        y_true_hours,
        pred_hours,
        quantiles=sorted(pred_hours.keys()),
        max_points=max_points,
        title=f"{dataset} Duration Quantile Intervals (Hours)",
        out_path=interval_dir / f"{dataset}_duration_candlestick_q10_q90.png",
    )

    plot_interval_width_histogram(
        y_true_hours,
        pred_hours,
        q_low=0.1,
        q_high=0.9,
        title=f"{dataset} Interval Widths in Hours",
        out_path=interval_dir / f"{dataset}_duration_interval_widths_q10_q90.png",
    )

    report = compute_quantile_interval_report(
        y_true_hours,
        pred_hours,
        interval_pairs=[(0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.4, 0.6), (0.5, 0.8)],
        quantile_checks=sorted(pred_hours.keys()),
        unit="hours",
    )
    save_quantile_interval_report(
        report,
        interval_dir / f"{dataset}_duration_quantile_interval_report.json",
    )

    # X_task = task_scaler.transform(merged[task_features])
    # merged["predicted_duration_task50"] = task_models[0.5].predict(X_task)
    # merged["predicted_duration_task80"] = task_models[0.8].predict(X_task)
    # merged["predicted_duration_task10"] = task_models[0.1].predict(X_task)
    # merged["predicted_duration_task"] = merged["predicted_duration_task50"]

    out_path = outputs_dir / "preprocessed" / "merged_with_duration_preds.csv"
    merged.to_csv(out_path, index=False)
    logger.info("Saved merged with duration predictions → %s", out_path)

def parse_args():
    ap = argparse.ArgumentParser(description="Event log pipeline (DB-free): preprocess and/or train sequence models.")
    ap.add_argument("--dataset", required=True, help="Folder name under DATA_DIR (or ./data).")
    ap.add_argument("--out", default="outputs", help="Output root directory")
    ap.add_argument("--preprocess", action="store_true", help="Run preprocessing only")
    ap.add_argument("--sequence", action="store_true", help="Train start/complete sequence models")
    ap.add_argument("--duration", action="store_true", help="Train duration classifiers and regressors")
    ap.add_argument("--verbose", action="store_true", help="Verbose console logs")
    ap.add_argument("--simulate-missing", action="store_true",
                    help="After preprocessing, create a version with induced missing values.")
    return ap.parse_args()

def main():
    args = parse_args()
    
    #Why it is here how can I make it simpler?
    output_root = Path(args.out).resolve()
    logger = setup_logger(output_root / "logs", verbose=args.verbose)
    merged_csv_path = None
    dataset = args.dataset
    
    
    
    if args.preprocess:
        merged_csv_path = step_preprocess(dataset, output_root, logger)

        if args.simulate_missing:
            step_simulate_missing(
                merged_csv_path,
                output_root,
                logger,
                target_col='start_date',
                missing_rate=0.15,
                random_state=42,
            )
    
    #Needs a big refactor later
    if args.sequence or args.duration:
        if merged_csv_path is None:
            merged_csv_path = output_root / "preprocessed" / f"{dataset}_merged.csv"
            if args.duration:
                merged_csv_path = output_root / "preprocessed" / f"{dataset}_case_seq_pred.csv"
            if not merged_csv_path.exists():
                raise FileNotFoundError(f"Expected {merged_csv_path}. Run with --preprocess first.")
        if args.sequence:
            step_train_sequence(merged_csv_path, dataset, output_root, logger)
        if args.duration:
            step_train_duration(merged_csv_path, dataset, output_root, logger)

    if not any([args.preprocess, args.simulate_missing, args.sequence, args.duration]):
        logger.warning("Nothing to do: pass --preprocess, --simulate_missing, --sequence, or --duration")
    else:
        logger.info("All requested stages completed.")


if __name__ == "__main__":
    main()

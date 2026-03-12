import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from eventlog_pipeline.io.loader import load_file
from eventlog_pipeline.preprocess.cleaning import clean_data
from eventlog_pipeline.preprocess.merge import merger
from eventlog_pipeline.preprocess.transform import (
    compute_durations_and_sequences,
    create_case_summary,
    prior_completions,
)
from eventlog_pipeline.preprocess.workload import compute_workload
from eventlog_pipeline.train.modeling import train_classifier_model, train_quantile_regressors
from eventlog_pipeline.train.plotting import (
    compute_quantile_interval_report,
    plot_interval_width_histogram,
    plot_prediction_intervals_candlestick,
    save_quantile_interval_report,
)
from eventlog_pipeline.train.predictors import (
    build_duration_class_predictors,
    build_duration_regression_predictors,
    build_sequence_predictors,
)
from eventlog_pipeline.utils.helpers import build_duration_classes_quantiles
from eventlog_pipeline.utils.logging import setup_logger


def _resolve_dataset_name(merged_csv: Path) -> str:
    dataset = merged_csv.stem
    for suffix in ("_merged", "_case_seq_pred"):
        if dataset.endswith(suffix):
            return dataset[: -len(suffix)]
    return dataset


def step_preprocess(dataset: str, outputs_dir: Path, logger) -> Path:
    logger.info("Loading dataset '%s'...", dataset)
    raw = load_file(dataset)
    logger.info("Raw rows: %d", len(raw))

    pre_dir = outputs_dir / "preprocessed"
    pre_dir.mkdir(parents=True, exist_ok=True)

    df = clean_data(raw)
    df = compute_durations_and_sequences(df)
    df, in_throughput_cols = prior_completions(df)

    queue_df, in_queue_cols, progress_df, in_progress_cols = compute_workload(df)
    case_summary_df, exists_cols = create_case_summary(df)

    case_summary_df.to_csv(pre_dir / f"{dataset}_case_summary.csv", index=False)

    merged, label_encoder = merger(df, queue_df, progress_df, case_summary_df)

    joblib.dump(label_encoder, pre_dir / f"{dataset}_task_label_encoder.joblib")
    joblib.dump(
        {
            "in_queue": in_queue_cols,
            "in_progress": in_progress_cols,
            "exists": exists_cols,
            "in_throughput": in_throughput_cols,
        },
        pre_dir / f"{dataset}_column_groups.joblib",
    )

    merged_path = pre_dir / f"{dataset}_merged.csv"
    merged.to_csv(merged_path, index=False)
    logger.info("Saved merged CSV %s", merged_path)
    return merged_path


def step_train_sequence(merged_csv: Path, outputs_dir: Path, logger) -> Path:
    logger.info("Loading merged data: %s", merged_csv)
    merged_csv = Path(merged_csv)
    dataset = _resolve_dataset_name(merged_csv)
    merged = pd.read_csv(merged_csv)

    # Less than one year total duration cases.
    merged = merged[merged["total_duration_sec"] < 31556926].copy()

    case_ids = shuffle(merged["case_id"].drop_duplicates(), random_state=42)
    num_total = len(case_ids)
    size_25pct = int(0.25 * num_total)

    case_ids_seq = case_ids.iloc[:size_25pct]
    case_ids_seq_pred = case_ids.iloc[size_25pct:]

    case_seq = merged[merged["case_id"].isin(case_ids_seq)].copy()
    case_seq_pred = merged[merged["case_id"].isin(case_ids_seq_pred)].copy()

    for col in ["relative_start", "relative_complete"]:
        if col in case_seq.columns:
            case_seq[col] = pd.to_numeric(case_seq[col], errors="coerce").fillna(0)
            case_seq[col] = np.log1p(np.maximum(case_seq[col], 0))

    col_groups = joblib.load(merged_csv.parent / f"{dataset}_column_groups.joblib")
    in_queue_cols = [c for c in col_groups["in_queue"] if c in merged.columns]
    in_progress_cols = [c for c in col_groups["in_progress"] if c in merged.columns]
    exists_cols = [c for c in col_groups["exists"] if c in merged.columns]
    in_throughput_cols = [c for c in col_groups["in_throughput"] if c in merged.columns]

    seq_predictors = build_sequence_predictors(
        in_progress_cols, in_queue_cols, exists_cols, in_throughput_cols
    )

    models_dir = outputs_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Train two separate models for start and complete sequence positions.
    start_models, start_scaler, start_features, start_metrics = train_quantile_regressors(
        case_seq[[*seq_predictors, "relative_start"]].dropna(subset=["relative_start"]),
        dataset,
        seq_predictors,
        label="relative_start",
        out_dir=models_dir,
        quantiles=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    )

    comp_models, comp_scaler, comp_features, comp_metrics = train_quantile_regressors(
        case_seq[[*seq_predictors, "relative_complete"]].dropna(subset=["relative_complete"]),
        dataset,
        seq_predictors,
        label="relative_complete",
        out_dir=models_dir,
        quantiles=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    )

    # Predict on held-out cases so duration classifiers can consume sequence predictions.
    x_pred_start = case_seq_pred[start_features]
    x_pred_start_scaled = start_scaler.transform(x_pred_start)
    case_seq_pred["predicted_start_sequence"] = start_models[0.5].predict(x_pred_start_scaled)

    x_pred_complete = case_seq_pred[comp_features]
    x_pred_complete_scaled = comp_scaler.transform(x_pred_complete)
    case_seq_pred["predicted_complete_sequence"] = comp_models[0.5].predict(
        x_pred_complete_scaled
    )

    reports_dir = outputs_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            f"{dataset}_sequence_start": start_metrics,
            f"{dataset}_sequence_complete": comp_metrics,
        },
        reports_dir / f"{dataset}_metrics_summary.joblib",
    )

    out_csv = outputs_dir / "preprocessed" / f"{dataset}_case_seq_pred.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    case_seq_pred.to_csv(out_csv, index=False)
    logger.info("Saved case sequence predictions -> %s", out_csv)
    return out_csv


def classify_binary_duration(v: float, zero_threshold: float = 1) -> int:
    return 0 if v <= zero_threshold else 1


def step_train_duration(merged_csv: Path, outputs_dir: Path, logger) -> Path:
    logger.info("Loading merged data for duration models: %s", merged_csv)
    merged_csv = Path(merged_csv)
    dataset = _resolve_dataset_name(merged_csv)
    merged = pd.read_csv(merged_csv)

    if "duration_sec" not in merged.columns:
        merged["duration"] = pd.to_timedelta(merged["duration"])
        merged["duration_sec"] = merged["duration"].dt.total_seconds().fillna(0)

    # Less than 6 months tasks.
    merged = merged[merged["duration_sec"] < 1.577e7].copy()

    case_ids = shuffle(merged["case_id"].drop_duplicates(), random_state=42)
    num_total = len(case_ids)
    size_25pct = int(0.25 * num_total)

    case_ids_bin_multi = case_ids.iloc[:size_25pct]
    case_ids_task_duration = case_ids.iloc[size_25pct:]

    df_bin_multi = merged[merged["case_id"].isin(case_ids_bin_multi)].copy()
    df_task_duration = merged[merged["case_id"].isin(case_ids_task_duration)].copy()

    for df in [df_bin_multi, df_task_duration]:
        if "duration_sec" in df.columns:
            df["duration_sec"] = pd.to_numeric(df["duration_sec"], errors="coerce").fillna(0)
            df["log_duration_sec"] = np.log1p(np.maximum(df["duration_sec"], 0))

    col_groups = joblib.load(merged_csv.parent / f"{dataset}_column_groups.joblib")
    in_queue_cols = [c for c in col_groups["in_queue"] if c in merged.columns]
    in_progress_cols = [c for c in col_groups["in_progress"] if c in merged.columns]
    exists_cols = [c for c in col_groups["exists"] if c in merged.columns]
    in_throughput_cols = [c for c in col_groups["in_throughput"] if c in merged.columns]

    ducl_predictors = build_duration_class_predictors(
        in_progress_cols, in_queue_cols, exists_cols, in_throughput_cols
    )
    du_predictors = build_duration_regression_predictors(
        in_progress_cols, in_queue_cols, exists_cols, in_throughput_cols
    )

    models_dir = outputs_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building duration classes from quantiles")

    df_bin_multi["binary_duration_class"] = df_bin_multi["duration_sec"].apply(
        classify_binary_duration
    )

    multi_labels, bin_edges = build_duration_classes_quantiles(
        df_bin_multi["duration_sec"],
        n_bins=5,
        zero_threshold=0.0,
    )
    df_bin_multi["multi_duration_class"] = multi_labels

    logger.info("Duration class bin edges (seconds): %s", bin_edges)
    joblib.dump({"bin_edges": bin_edges}, models_dir / f"{dataset}_duration_bins.joblib")

    logger.info("Training duration classifiers")

    if len(df_bin_multi["binary_duration_class"].unique()) == 2:
        bin_model, bin_scaler, bin_le = train_classifier_model(
            df_bin_multi,
            dataset,
            ducl_predictors,
            "binary_duration_class",
            out_dir=models_dir / "duration_binary",
        )
        x_all_bin = bin_scaler.transform(df_task_duration[ducl_predictors])
        df_task_duration["binary_pred"] = bin_le.inverse_transform(
            bin_model.predict(x_all_bin).astype(int)
        )
    else:
        logger.warning(
            "Skipping binary duration classifier training due to insufficient class variety."
        )
        df_task_duration["binary_pred"] = 1

    non_zero = df_bin_multi.query("binary_duration_class == 1")
    if non_zero["multi_duration_class"].nunique() > 1:
        multi_model, multi_scaler, multi_le = train_classifier_model(
            non_zero,
            dataset,
            ducl_predictors,
            "multi_duration_class",
            out_dir=models_dir / "duration_multi",
        )

        non_zero_mask = df_task_duration["binary_pred"] == 1
        x_all_multi = multi_scaler.transform(
            df_task_duration.loc[non_zero_mask, ducl_predictors]
        )
        df_task_duration.loc[non_zero_mask, "multi_pred"] = multi_le.inverse_transform(
            multi_model.predict(x_all_multi).astype(int)
        )
        df_task_duration.loc[~non_zero_mask, "multi_pred"] = 0
    else:
        logger.warning("Skipping multi duration classifier training due to insufficient variety.")
        df_task_duration["multi_pred"] = 0

    df_task_duration["predicted_duration_class"] = df_task_duration["multi_pred"].astype(int)

    logger.info("Training task duration quantile regressor on log duration")
    task_models, task_scaler, task_features, task_metrics = train_quantile_regressors(
        df_task_duration[[*du_predictors, "log_duration_sec"]],
        dataset,
        du_predictors,
        label="log_duration_sec",
        out_dir=models_dir / "task_duration",
        quantiles=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    )

    x_all = task_scaler.transform(df_task_duration[task_features])
    pred_log = {q: model.predict(x_all) for q, model in task_models.items()}

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

    out_path = outputs_dir / "preprocessed" / f"{dataset}_duration_predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_task_duration.to_csv(out_path, index=False)
    logger.info("Saved duration class/regression predictions -> %s", out_path)
    return out_path


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Event log pipeline: preprocess, sequence prediction, and duration "
            "binary/multiclass classification + regression."
        )
    )
    ap.add_argument("--dataset", required=True, help="Folder name under ./data")
    ap.add_argument("--out", default="outputs", help="Output root directory")
    ap.add_argument("--preprocess", action="store_true", help="Run preprocessing stage")
    ap.add_argument("--sequence", action="store_true", help="Train start/complete sequence models")
    ap.add_argument(
        "--duration",
        action="store_true",
        help="Train duration classifiers and quantile regression",
    )
    ap.add_argument("--verbose", action="store_true", help="Verbose console logs")
    return ap.parse_args()


def main():
    args = parse_args()

    output_root = Path(args.out).resolve()
    logger = setup_logger(output_root / "logs", verbose=args.verbose)
    dataset = args.dataset

    merged_csv_path = None

    if args.preprocess:
        merged_csv_path = step_preprocess(dataset, output_root, logger)

    if args.sequence:
        if merged_csv_path is None:
            merged_csv_path = output_root / "preprocessed" / f"{dataset}_merged.csv"
            if not merged_csv_path.exists():
                raise FileNotFoundError(f"Expected {merged_csv_path}. Run with --preprocess first.")
        merged_csv_path = step_train_sequence(merged_csv_path, output_root, logger)

    if args.duration:
        if merged_csv_path is None or merged_csv_path.name.endswith("_merged.csv"):
            merged_csv_path = output_root / "preprocessed" / f"{dataset}_case_seq_pred.csv"
            if not merged_csv_path.exists():
                raise FileNotFoundError(
                    f"Expected {merged_csv_path}. Run with --sequence first."
                )
        step_train_duration(merged_csv_path, output_root, logger)

    if not any([args.preprocess, args.sequence, args.duration]):
        logger.warning("Nothing to do: pass --preprocess, --sequence, and/or --duration")
    else:
        logger.info("All requested stages completed.")


if __name__ == "__main__":
    main()

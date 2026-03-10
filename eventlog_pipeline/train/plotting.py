from pathlib import Path
import json
import textwrap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd

def _save(plt_obj, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt_obj.savefig(out_path, bbox_inches='tight', dpi=160)
    plt_obj.close()

def _set_plot_style():
    sns.set_theme(style="whitegrid")

def _normalize_importances(importances):
    values = np.asarray(importances, dtype=float)
    total = float(np.sum(values))
    if total <= 0:
        return values
    return values / total

def _format_feature_label(feature, pretty_labels=True, wrap_width=40):
    label = feature.replace("_", " ") if pretty_labels else feature
    if wrap_width:
        return textwrap.fill(label, width=wrap_width)
    return label

def plot_learning_curve_from_model(model, out_path: Path):
    results = model.evals_result()
    if not results:
        return
    metric = list(results['validation_0'].keys())[0]
    train_metric = results['validation_0'][metric]
    test_metric = results['validation_1'][metric] if 'validation_1' in results else None
    _set_plot_style()
    plt.figure(figsize=(8, 5))
    plt.plot(train_metric, label='Training Loss')
    if test_metric is not None:
        plt.plot(test_metric, label='Validation Loss', linestyle='--')
    plt.xlabel('Iterations')
    plt.ylabel(metric.upper())
    plt.title('Learning Curve')
    plt.legend()
    _save(plt, out_path)

def plot_predictions_scatter(y_true, y_pred, label, title, out_path: Path):
    

    _set_plot_style()
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    lo = float(min(min(y_true), min(y_pred)))
    hi = float(max(max(y_true), max(y_pred)))
    plt.plot([lo, hi], [lo, hi], linestyle='--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    _save(plt, out_path)

def plot_residuals_scatter(y_true, y_pred, title, out_path: Path):
    residuals = np.array(y_true) - np.array(y_pred)
    _set_plot_style()
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals (True - Predicted)')
    plt.title(title)
    _save(plt, out_path)

def save_feature_importance_bar(
    importances,
    features,
    title,
    out_path: Path,
    top_k=10,
    base_features=None,
    pretty_labels=True,
    wrap_width=40,
):
    df = pd.DataFrame({'Feature': features, 'Importance': importances})
    df = df.sort_values('Importance', ascending=False).head(top_k)
    df["FeatureLabel"] = df["Feature"].apply(
        lambda name: _format_feature_label(name, pretty_labels, wrap_width)
    )
    if base_features is not None:
        base_set = set(base_features)
        df["FeatureGroup"] = df["Feature"].apply(
            lambda name: "base" if name in base_set else "additional"
        )
        palette = {"base": "#4C78A8", "additional": "#F28E2B"}
    else:
        df["FeatureGroup"] = "feature"
        palette = {"feature": "#4C78A8"}

    _set_plot_style()
    height = max(4.0, 0.45 * len(df) + 1.5)
    fig, ax = plt.subplots(figsize=(10, height))
    sns.barplot(
        x="Importance",
        y="FeatureLabel",
        data=df,
        hue="FeatureGroup",
        dodge=False,
        palette=palette,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    sns.despine(ax=ax, left=True)
    if base_features is not None:
        ax.legend(title="Feature Group", loc="lower right")
    else:
        ax.get_legend().remove()
    _save(plt, out_path)

def save_feature_importance_comparison(
    importances_a,
    features_a,
    importances_b,
    features_b,
    title,
    out_path: Path,
    dataset_a="Dataset A",
    dataset_b="Dataset B",
    base_features=None,
    top_k=15,
    normalize=False,
    pretty_labels=True,
    wrap_width=40,
):
    imp_a = _normalize_importances(importances_a) if normalize else np.asarray(importances_a, dtype=float)
    imp_b = _normalize_importances(importances_b) if normalize else np.asarray(importances_b, dtype=float)
    df_a = pd.DataFrame({"Feature": features_a, "Importance": imp_a, "Dataset": dataset_a})
    df_b = pd.DataFrame({"Feature": features_b, "Importance": imp_b, "Dataset": dataset_b})
    df = pd.concat([df_a, df_b], ignore_index=True)

    max_importance = df.groupby("Feature")["Importance"].max().sort_values(ascending=False)
    top_features = list(max_importance.head(top_k).index)
    datasets = [dataset_a, dataset_b]

    full_index = pd.MultiIndex.from_product([top_features, datasets], names=["Feature", "Dataset"])
    df_plot = df.set_index(["Feature", "Dataset"]).reindex(full_index, fill_value=0).reset_index()
    df_plot["FeatureLabel"] = df_plot["Feature"].apply(
        lambda name: _format_feature_label(name, pretty_labels, wrap_width)
    )
    order = [_format_feature_label(name, pretty_labels, wrap_width) for name in top_features]

    _set_plot_style()
    height = max(4.5, 0.5 * len(top_features) + 1.5)
    fig, ax = plt.subplots(figsize=(11, height))
    sns.barplot(
        data=df_plot,
        x="Importance",
        y="FeatureLabel",
        hue="Dataset",
        order=order,
        palette="tab10",
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Importance (normalized)" if normalize else "Importance")
    ax.set_ylabel("")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    sns.despine(ax=ax, left=True)

    if base_features is not None:
        base_set = set(base_features)
        group_palette = {"base": "#4C78A8", "additional": "#F28E2B"}
        group_by_label = {
            _format_feature_label(name, pretty_labels, wrap_width):
            ("base" if name in base_set else "additional")
            for name in top_features
        }
        for tick in ax.get_yticklabels():
            group = group_by_label.get(tick.get_text(), "additional")
            tick.set_color(group_palette[group])
        dataset_handles, dataset_labels = ax.get_legend_handles_labels()
        dataset_legend = ax.legend(
            dataset_handles, dataset_labels, title="Dataset", loc="upper right"
        )
        ax.add_artist(dataset_legend)
        group_handles = [
            mpatches.Patch(color=group_palette["base"], label="Base features"),
            mpatches.Patch(color=group_palette["additional"], label="Additional features"),
        ]
        ax.legend(handles=group_handles, title="Feature Group", loc="lower right")
    else:
        ax.legend(title="Dataset", loc="lower right")

    _save(plt, out_path)

# def plot_hist(series, bins=50, title="", out_path: Path = None, annotate_mean=False):
#     import numpy as np
#     plt.figure(figsize=(10, 6))
#     values = np.array(series.dropna())
#     plt.hist(values, bins=bins, edgecolor='black')
#     if annotate_mean:
#         mean_value = float(np.mean(np.abs(values)))
#         plt.text(mean_value * 1.05, plt.gca().get_ylim()[1] * 0.8, f'Mean: {mean_value:.2f}')
#         plt.axvline(x=mean_value, linestyle='--')
#     plt.title(title)
#     plt.xlabel('Value')
#     plt.ylabel('Frequency')
#     if out_path:
#         _save(plt, out_path)
#     else:
#         plt.show()

# def plot_prediction_intervals_candlestick(
#     df,
#     lower_bound_col,
#     upper_bound_col,
#     prediction_col,
#     true_value_col,
#     lower_true_value_threshold=20,
#     upper_true_value_threshold=2000,
#     upper_bound_threshold=400,
#     sample_percent=0.1,
#     title="Prediction Intervals",
#     out_path: Path = None,
# ):
#     import numpy as np
#     mask = (
#         (df[true_value_col] >= lower_true_value_threshold) &
#         (df[true_value_col] <= upper_true_value_threshold) &
#         (df[upper_bound_col] < upper_bound_threshold)
#     )
#     filtered_df = df.loc[mask].copy()
#     n_samples = len(filtered_df)
#     sample_size = max(1, int(n_samples * sample_percent / 100.0))
#     sampled_df = filtered_df.sample(n=sample_size, random_state=42)

#     true_values = sampled_df[true_value_col].values
#     median_predictions = sampled_df[prediction_col].values
#     lower_bound = sampled_df[lower_bound_col].values
#     upper_bound = sampled_df[upper_bound_col].values

#     plt.figure(figsize=(12, 6))
#     idx = range(len(true_values))
#     plt.vlines(idx, lower_bound, upper_bound, alpha=0.5, linewidth=1, label='Prediction Interval')
#     plt.scatter(idx, median_predictions, marker='_', s=100, linewidth=2, label='Median Prediction')
#     plt.scatter(idx, true_values, marker='o', label='True Values')

#     yerr_lower = np.abs(median_predictions - lower_bound)
#     yerr_upper = np.abs(upper_bound - median_predictions)
#     plt.errorbar(idx, median_predictions, yerr=[yerr_lower, yerr_upper], fmt='none', alpha=0.3, capsize=5)

#     plt.xlabel('Sample Index')
#     plt.ylabel('Task/Application Duration')
#     plt.title(title)
#     plt.legend()

#     if out_path:
#         _save(plt, out_path)
#     else:
#         plt.show()

#     return sampled_df

# def plot_duration_predictions(
#     df: pd.DataFrame,
#     true_col: str,
#     pred_cols: list[str],
#     out_path: Path,
#     title: str = "Task Duration Predictions",
#     sample_frac: float = 0.05,
# ):
#     """
#     Plot true vs predicted duration values for different quantile predictions.
#     pred_cols: list of predicted duration columns, e.g. ['predicted_duration_task10', 'predicted_duration_task50', 'predicted_duration_task80']
#     """
    

#     if sample_frac < 1.0:
#         df = df.sample(frac=sample_frac, random_state=42)

#     plt.figure(figsize=(10, 6))
#     plt.scatter(df.index, df[true_col], label="True Duration", alpha=0.5, s=25)

#     for col in pred_cols:
#         if col in df.columns:
#             plt.scatter(df.index, df[col], label=col.replace("predicted_", "").replace("_task", ""), alpha=0.6, s=20)

#     plt.xlabel("Sample Index")
#     plt.ylabel("Log Duration")
#     plt.title(title)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=160, bbox_inches="tight")
#     plt.close()
#     print(f"[PLOT] Saved duration predictions → {out_path}")


def plot_duration_class_comparison(
    y_true: pd.Series,
    y_pred: pd.Series,
    out_path: Path,
    title: str = "Duration Class Prediction Comparison",
    label_encoder=None,
):

    # Inverse transform if encoder provided
    if label_encoder is not None:
        y_true_display = label_encoder.inverse_transform(y_true.astype(int))
        y_pred_display = label_encoder.inverse_transform(y_pred.astype(int))
    else:
        y_true_display = y_true.astype(int)
        y_pred_display = y_pred.astype(int)
    
    df = pd.DataFrame({"True": y_true_display, "Pred": y_pred_display})
    cm = pd.crosstab(df["True"], df["Pred"], normalize="index")
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=True)
    plt.title(title)
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved class comparison heatmap → {out_path}")



def _align_quantile_predictions(y_true, predictions):
    y = np.asarray(y_true, dtype=float)
    preds = {float(q): np.asarray(v, dtype=float) for q, v in predictions.items()}
    if not preds:
        raise ValueError("No quantile predictions provided.")
    n = len(y)
    for q, arr in preds.items():
        if len(arr) != n:
            raise ValueError(f"Quantile {q} length {len(arr)} does not match y_true length {n}.")
    mask = np.isfinite(y)
    for arr in preds.values():
        mask &= np.isfinite(arr)
    y = y[mask]
    preds = {q: arr[mask] for q, arr in preds.items()}
    return y, preds


def _choose_sample_indices(n_samples, max_points):
    if n_samples <= 0:
        return np.array([], dtype=int)
    if max_points is None or max_points <= 0:
        return np.arange(n_samples)
    if n_samples <= max_points:
        return np.arange(n_samples)
    return np.linspace(0, n_samples - 1, num=max_points, dtype=int)


def plot_prediction_intervals_candlestick(
    y_true,
    predictions,
    quantiles=(0.1, 0.5, 0.9),
    max_points=300,
    title="Prediction Intervals (Candlestick)",
    out_path: Path = None,
):
    """Candlestick-style plot of quantile prediction intervals vs ground truth."""
    y, preds = _align_quantile_predictions(y_true, predictions)
    available = sorted(preds.keys())
    if quantiles is None:
        quantiles = available
    quantiles = [float(q) for q in quantiles if float(q) in preds]
    if len(quantiles) < 2:
        raise ValueError("Need at least two quantiles to plot intervals.")

    sorted_q = sorted(set(quantiles))
    lower_q = sorted_q[0]
    upper_q = sorted_q[-1]
    if 0.5 in preds:
        median_q = 0.5
    else:
        median_q = sorted_q[len(sorted_q) // 2]

    lower = preds[lower_q]
    upper = preds[upper_q]
    median = preds[median_q]

    sample_idx = _choose_sample_indices(len(y), max_points)
    y_s = y[sample_idx]
    lower_s = lower[sample_idx]
    upper_s = upper[sample_idx]
    median_s = median[sample_idx]

    _set_plot_style()
    plt.figure(figsize=(12, 6))
    x = np.arange(len(sample_idx))
    plt.vlines(x, lower_s, upper_s, color="#D55E00", alpha=0.45, linewidth=1, label=f"Interval q{lower_q}-q{upper_q}")
    plt.scatter(x, median_s, color="#D55E00", marker="_", s=100, linewidth=2, label=f"Median q{median_q}")
    plt.scatter(x, y_s, color="#1F77B4", marker="o", s=22, alpha=0.7, label="Ground Truth")

    yerr_lower = np.abs(median_s - lower_s)
    yerr_upper = np.abs(upper_s - median_s)
    plt.errorbar(
        x,
        median_s,
        yerr=[yerr_lower, yerr_upper],
        fmt="none",
        ecolor="#D55E00",
        alpha=0.25,
        capsize=3,
    )

    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if out_path is not None:
        _save(plt, out_path)
    else:
        plt.show()

    return pd.DataFrame(
        {
            "sample_index": sample_idx,
            "y_true": y_s,
            f"q{lower_q}": lower_s,
            f"q{median_q}": median_s,
            f"q{upper_q}": upper_s,
            "interval_width": upper_s - lower_s,
        }
    )


def plot_interval_width_histogram(
    y_true,
    predictions,
    q_low=0.1,
    q_high=0.9,
    bins=50,
    title="Prediction Interval Widths",
    out_path: Path = None,
):
    y, preds = _align_quantile_predictions(y_true, predictions)
    q_low = float(q_low)
    q_high = float(q_high)
    if q_low not in preds or q_high not in preds:
        raise ValueError(f"Quantiles {q_low} and {q_high} not found in predictions.")
    widths = preds[q_high] - preds[q_low]
    widths = widths[np.isfinite(widths)]
    if len(widths) == 0:
        raise ValueError("No valid interval widths to plot.")

    _set_plot_style()
    plt.figure(figsize=(9, 5))
    plt.hist(widths, bins=bins, edgecolor="black", alpha=0.7)
    mean_width = float(np.mean(widths))
    median_width = float(np.median(widths))
    plt.axvline(mean_width, color="black", linestyle="--", linewidth=1.2, label=f"Mean: {mean_width:.2f}")
    plt.axvline(median_width, color="gray", linestyle=":", linewidth=1.2, label=f"Median: {median_width:.2f}")
    plt.xlabel("Interval Width")
    plt.ylabel("Count")
    plt.title(f"{title} (q{q_low}-q{q_high})")
    plt.legend()
    plt.tight_layout()

    if out_path is not None:
        _save(plt, out_path)
    else:
        plt.show()

    return widths


def compute_quantile_interval_report(
    y_true,
    predictions,
    interval_pairs=None,
    quantile_checks=None,
    unit="value",
):
    y, preds = _align_quantile_predictions(y_true, predictions)
    available = sorted(preds.keys())

    if interval_pairs is None:
        interval_pairs = [(0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.4, 0.6)]
    interval_pairs = [
        (float(a), float(b)) for a, b in interval_pairs if float(a) in preds and float(b) in preds
    ]

    if quantile_checks is None:
        quantile_checks = available
    quantile_checks = [float(q) for q in quantile_checks if float(q) in preds]

    interval_rows = []
    for q_low, q_high in interval_pairs:
        lower = preds[q_low]
        upper = preds[q_high]
        inside = (y >= lower) & (y <= upper)
        below = y < lower
        above = y > upper
        widths = upper - lower
        crossings = upper < lower
        widths_nonneg = np.maximum(widths, 0)

        row = {
            "range": f"{q_low}-{q_high}",
            "q_low": q_low,
            "q_high": q_high,
            "coverage_pct": float(inside.mean() * 100),
            "below_pct": float(below.mean() * 100),
            "above_pct": float(above.mean() * 100),
            "expected_coverage_pct": float((q_high - q_low) * 100),
            "coverage_error_pct": float((inside.mean() - (q_high - q_low)) * 100),
            "crossing_pct": float(crossings.mean() * 100),
            "mean_width": float(np.mean(widths_nonneg)),
            "median_width": float(np.median(widths_nonneg)),
            "p10_width": float(np.percentile(widths_nonneg, 10)),
            "p90_width": float(np.percentile(widths_nonneg, 90)),
        }
        interval_rows.append(row)

    quantile_rows = []
    for q in quantile_checks:
        pred = preds[q]
        pct_below = float((y <= pred).mean() * 100)
        quantile_rows.append(
            {
                "quantile": q,
                "pct_true_below_or_equal": pct_below,
                "pct_true_above": float(100 - pct_below),
                "expected_pct": float(q * 100),
                "error_pct": float(pct_below - q * 100),
            }
        )

    report = {
        "n_samples": int(len(y)),
        "unit": unit,
        "available_quantiles": available,
        "interval_coverage": interval_rows,
        "quantile_coverage": quantile_rows,
    }
    return report


def save_quantile_interval_report(report: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

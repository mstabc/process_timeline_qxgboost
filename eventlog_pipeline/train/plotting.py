from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def _save(plt_obj, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt_obj.savefig(out_path, bbox_inches='tight', dpi=160)
    plt_obj.close()

def plot_learning_curve_from_model(model, out_path: Path):
    results = model.evals_result()
    if not results:
        return
    metric = list(results['validation_0'].keys())[0]
    train_metric = results['validation_0'][metric]
    test_metric = results['validation_1'][metric] if 'validation_1' in results else None
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
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals (True - Predicted)')
    plt.title(title)
    _save(plt, out_path)

def save_feature_importance_bar(importances, features, title, out_path: Path, top_k=10):

    df = pd.DataFrame({'Feature': features, 'Importance': importances}) \
           .sort_values('Importance', ascending=False).head(top_k)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=df)
    plt.title(title)
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
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
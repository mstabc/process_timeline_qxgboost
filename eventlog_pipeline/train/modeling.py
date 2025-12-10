from pathlib import Path
from typing import Dict, List, Tuple


import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder


from xgboost import XGBClassifier
import joblib
from xgboost import XGBRegressor
import json


from .plotting import (
    plot_learning_curve_from_model,
    plot_predictions_scatter,
    plot_residuals_scatter,
    save_feature_importance_bar, plot_duration_class_comparison
)


def train_quantile_regressors(
    df: pd.DataFrame,
    dataset: str,
    predictors: List[str],
    label: str,
    out_dir: Path,
    quantiles=(0.5, 0.6),
) -> Tuple[Dict[float, "XGBRegressor"], StandardScaler, List[str], dict]:


    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    
    predictors = [c for c in predictors if c in df.columns]

    y = pd.to_numeric(df[label], errors="coerce")
    X = df[predictors]
    keep = y.notna()
    X, y = X.loc[keep], y.loc[keep]

    if X.shape[1] == 0 or y.isna().all():
        raise ValueError(f"No valid data for label {label}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    inversed = 0
    models, metrics = {}, {}
    for q in quantiles:
        model = XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=float(q),
            n_estimators=500,
            learning_rate=0.01,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric="mae",
        )
        model.fit(X_train_s, y_train, eval_set=[(X_test_s, y_test)], verbose=False)
        y_pred = model.predict(X_test_s)
        if label == "duration_sec":
            if  inversed == 0:    
                y_test = np.expm1(y_test) / 3600
                inversed = 1
            y_pred = np.expm1(y_pred) / 3600
        metrics[str(q)] = {
            "MSE": float(((y_test - y_pred) ** 2).mean()),
            "MAE": float(np.mean(np.abs(y_test - y_pred))),
        }
        print(f"{label} q={q} - MSE: {metrics[str(q)]['MSE']:.4f}, MAE: {metrics[str(q)]['MAE']:.4f}")
        
        plot_learning_curve_from_model(model, plots_dir / f"{dataset}_{label}_q{q}_learning.png")
        plot_predictions_scatter(y_test, y_pred, label, f"{label} q={q} Actual vs Pred", plots_dir / f"{dataset}_{label}_q{q}_actual_vs_pred.png")
        plot_residuals_scatter(y_test, y_pred, f"{label} q={q} Residuals", plots_dir / f"{dataset}_{label}_q{q}_residuals.png")

        joblib.dump(model, out_dir / f"{dataset}_{label}_q{q}.joblib")
        models[q] = model

    # save feature importance
    importances = models[0.5].feature_importances_
    save_feature_importance_bar(importances, list(X.columns), f"{label} - Top Features", plots_dir / f"{dataset}_{label}_feature_importance.png", top_k=15)

    

    # save metadata
    joblib.dump(scaler, out_dir / f"{dataset}_{label}_scaler.joblib")
    joblib.dump(list(X.columns), out_dir / f"{dataset}_{label}_features.joblib")
    with open(out_dir / f"{dataset}_{label}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    txt_path = out_dir / f"{dataset}_{label}_metrics.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Label: {label}\n\n")
        for q, m in metrics.items():
            f.write(
                f"Quantile {q}:\n"
                f"  MSE: {m['MSE']:.4f}\n"
                f"  MAE: {m['MAE']:.4f}\n\n"
            )

    print(f"[INFO] Saved quantile regression plots → {plots_dir}")
    return models, scaler, list(X.columns), metrics


def train_classifier_model(
    df: pd.DataFrame,
    dataset: str,
    predictors: List[str],
    label: str,
    out_dir: Path
) -> Tuple["XGBClassifier", StandardScaler, dict]:
    

    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    predictors = [c for c in predictors if c in df.columns]

    X = df[predictors]
    y = pd.to_numeric(df[label], errors="coerce").fillna(0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    num_classes = len(le.classes_)
    objective = "binary:logistic" if num_classes == 2 else "multi:softprob"


    model = XGBClassifier(
        objective=objective,
        n_estimators=800,
        learning_rate=0.01,
        max_depth=6,
        colsample_bytree=0.8,
        random_state=42,
        **({"num_class": num_classes} if num_classes > 2 else {}),
    )

    model.fit(X_train_s, y_train_enc, eval_set=[(X_test_s, y_test_enc)], verbose=False)
    y_pred_enc = model.predict(X_test_s)

    importances = model.feature_importances_
    save_feature_importance_bar(importances, list(X.columns), f"{dataset} - {label} - Top Features", plots_dir / f"{dataset}_{label}_Classification_feature_importance.png", top_k=15)
    
    plot_learning_curve_from_model(model, plots_dir / f"{dataset}_{label}_classification_learning.png")
    plot_duration_class_comparison(
    y_test_enc, 
    y_pred_enc, 
    plots_dir / f"{dataset}_{label}_class_comparison.png",
    title="Duration Class Prediction (Original Labels)",
    label_encoder=le
)
    
    acc = float(accuracy_score(y_test_enc, y_pred_enc))
    print(f"{label} classifier accuracy: {acc:.4f}")
    
    cm = confusion_matrix(y_test_enc, y_pred_enc)
    print(f"{label} classifier confusion matrix:\n{cm}")
    txt_path = out_dir / f"{dataset}_{label}_classifier_metrics.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Label: {label}\n\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Confusion matrix (rows: true, columns: predicted):\n")
        f.write(np.array2string(cm))
        f.write("\n\nLabel encoding:\n")
        for idx, cls in enumerate(le.classes_):
            f.write(f"  {idx}: {cls}\n")
    
    joblib.dump(model, out_dir / f"{dataset} - {label}_classifier.joblib")
    joblib.dump(scaler, out_dir / f"{dataset} - {label}_classifier_scaler.joblib")
    joblib.dump(le, out_dir / f"{dataset} - {label}_classifier_label_encoder.joblib")

    print(f"[INFO] Saved classifier plots → {plots_dir}")
    return model, scaler, le

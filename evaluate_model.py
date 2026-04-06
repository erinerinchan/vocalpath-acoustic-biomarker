"""
Standalone model evaluation script for VocalPath.

Loads the trained model and evaluates it on either real or synthetic data,
printing a comprehensive report with per-class metrics, confusion matrix,
bootstrap CIs, and threshold analysis.

Usage:
    python evaluate_model.py                   # auto-detect best data source
    python evaluate_model.py --data synthetic  # force synthetic data
    python evaluate_model.py --data real       # force real clinical data
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def load_data(source="auto"):
    """Load feature data from the specified source."""
    real_path = "data/real_features.csv"
    synth_path = "data/features.csv"

    if source == "real":
        if not os.path.exists(real_path):
            raise FileNotFoundError(f"Real data not found at {real_path}")
        df = pd.read_csv(real_path)
        name = "VOICED clinical dataset"
    elif source == "synthetic":
        df = pd.read_csv(synth_path)
        name = "Synthetic dataset"
    else:  # auto
        if os.path.exists(real_path):
            df = pd.read_csv(real_path)
            name = "VOICED clinical dataset"
        else:
            df = pd.read_csv(synth_path)
            name = "Synthetic dataset"

    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values
    y = df["label"].values
    return X, y, feature_cols, name


def bootstrap_ci(y_true, y_pred, y_proba, n_boot=1000, seed=42):
    """Compute 95% bootstrap confidence intervals for key metrics."""
    rng = np.random.RandomState(seed)
    results = {m: [] for m in ["accuracy", "f1", "recall", "precision", "roc_auc"]}

    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        results["accuracy"].append(accuracy_score(y_true[idx], y_pred[idx]))
        results["f1"].append(f1_score(y_true[idx], y_pred[idx]))
        results["recall"].append(recall_score(y_true[idx], y_pred[idx]))
        results["precision"].append(precision_score(y_true[idx], y_pred[idx]))
        results["roc_auc"].append(roc_auc_score(y_true[idx], y_proba[idx]))

    ci = {}
    for metric, values in results.items():
        low, high = np.percentile(values, [2.5, 97.5])
        ci[metric] = {"mean": np.mean(values), "ci_low": low, "ci_high": high}
    return ci


def threshold_analysis(y_true, y_proba):
    """Find optimal thresholds for different recall targets."""
    thresholds = np.arange(0.1, 0.91, 0.05)
    print("\nThreshold Analysis:")
    print(f"{'Threshold':>10} {'Recall':>8} {'Precision':>10} {'F1':>8} {'Accuracy':>10}")
    print("-" * 50)
    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        r = recall_score(y_true, y_pred_t)
        p = precision_score(y_true, y_pred_t, zero_division=0)
        f = f1_score(y_true, y_pred_t)
        a = accuracy_score(y_true, y_pred_t)
        marker = " <-- default" if abs(t - 0.5) < 0.01 else ""
        print(f"{t:>10.2f} {r:>8.3f} {p:>10.3f} {f:>8.3f} {a:>10.3f}{marker}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate VocalPath model")
    parser.add_argument("--data", choices=["auto", "real", "synthetic"],
                        default="auto", help="Data source to evaluate on")
    args = parser.parse_args()

    # Load model
    model = joblib.load("model/rf_classifier.joblib")
    feature_names = joblib.load("model/feature_names.joblib")

    # Load data
    X, y, feature_cols, data_name = load_data(args.data)
    print(f"Model Evaluation Report")
    print(f"{'=' * 60}")
    print(f"Data source: {data_name}")
    print(f"Samples: {len(y)} (Healthy={sum(y == 0)}, Pathological={sum(y == 1)})")
    print(f"Features: {len(feature_cols)}")

    # Cross-validated predictions
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X, y, cv=cv)
    y_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]

    # Classification report
    print(f"\nClassification Report (5-fold CV):")
    print(classification_report(y, y_pred, target_names=["Healthy", "Pathological"]))

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    print(f"Confusion Matrix:")
    print(f"                  Pred Healthy  Pred Pathological")
    print(f"  Actual Healthy      {cm[0, 0]:>5d}          {cm[0, 1]:>5d}")
    print(f"  Actual Pathological {cm[1, 0]:>5d}          {cm[1, 1]:>5d}")

    # Bootstrap CIs
    y_pred_binary = (y_proba >= 0.5).astype(int)
    ci = bootstrap_ci(y, y_pred_binary, y_proba)
    print(f"\n95% Bootstrap Confidence Intervals (1000 iterations):")
    for metric, vals in ci.items():
        print(f"  {metric:>12s}: {vals['mean']:.3f}  [{vals['ci_low']:.3f}, {vals['ci_high']:.3f}]")

    # Threshold analysis
    threshold_analysis(y, y_proba)

    # Save evaluation report
    report = {
        "data_source": data_name,
        "n_samples": int(len(y)),
        "n_healthy": int(sum(y == 0)),
        "n_pathological": int(sum(y == 1)),
        "cv_accuracy": float(accuracy_score(y, y_pred)),
        "cv_f1": float(f1_score(y, y_pred)),
        "cv_recall": float(recall_score(y, y_pred)),
        "cv_precision": float(precision_score(y, y_pred)),
        "cv_auc_roc": float(roc_auc_score(y, y_proba)),
        "bootstrap_ci": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in ci.items()},
    }
    with open("model/evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved evaluation report to model/evaluation_report.json")


if __name__ == "__main__":
    main()

"""
Train and evaluate multiple classifiers on extracted voice features.

What this script does (plain English):
    1. Loads the synthetic voice data created by generate_demo_data.py
    2. Trains three different machine-learning algorithms on that data
    3. Tests each one using 5-fold cross-validation (explained below)
    4. Picks the winner (highest F1 score) and saves it to disk
    5. Also saves evaluation charts (confusion matrix, ROC curve,
       feature importance) so the Streamlit app can display them

    Cross-validation means: split the data into 5 chunks, train on 4,
    test on the 1 the model hasn't seen, rotate through all 5.  This
    prevents the model from just memorising the training answers.

Outputs saved to model/:
    rf_classifier.joblib      – the trained model (ready for predictions)
    feature_names.joblib      – list of the 26 feature column names
    metrics.json              – accuracy, precision, recall, F1, AUC for all 3 models
    confusion_matrix.csv      – how many predictions were right vs wrong
    roc_data.csv              – data for the ROC curve plot
    feature_importance.csv    – which voice measurements matter most
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import (StratifiedKFold, cross_validate, cross_val_predict,
                                     train_test_split)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (confusion_matrix, roc_curve, auc, classification_report,
                             precision_recall_curve, average_precision_score,
                             accuracy_score, f1_score, recall_score, precision_score,
                             roc_auc_score)
from sklearn.model_selection import learning_curve
import joblib


def evaluate_model(pipeline, X, y, cv):
    """
    Run 5-fold cross-validation and return average scores.

    We measure five things:
      - accuracy:  % of all predictions that were correct
      - precision: when we say "pathological", how often are we right?
      - recall:    of all truly pathological voices, how many did we catch?
      - f1:        a single number balancing precision and recall
      - roc_auc:   overall ability to separate healthy from pathological
    """
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring)
    return {
        "accuracy": cv_results["test_accuracy"].mean(),
        "accuracy_std": cv_results["test_accuracy"].std(),
        "precision": cv_results["test_precision"].mean(),
        "precision_std": cv_results["test_precision"].std(),
        "recall": cv_results["test_recall"].mean(),
        "recall_std": cv_results["test_recall"].std(),
        "f1": cv_results["test_f1"].mean(),
        "f1_std": cv_results["test_f1"].std(),
        "auc_roc": cv_results["test_roc_auc"].mean(),
        "auc_roc_std": cv_results["test_roc_auc"].std(),
    }


def main():
    # ── 1. Load data ──
    # Prefer real clinical data if available; fall back to synthetic
    real_path = "data/real_features.csv"
    synth_path = "data/features.csv"

    if os.path.exists(real_path):
        df = pd.read_csv(real_path)
        data_source = "VOICED clinical dataset (Cesari et al., 2018)"
    else:
        df = pd.read_csv(synth_path)
        data_source = "Synthetic (generate_demo_data.py)"

    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values   # the 26 voice measurements (input)
    y = df["label"].values         # 0 = healthy, 1 = pathological (answer key)

    print(f"Data source: {data_source}")
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: Healthy={sum(y == 0)}, Pathological={sum(y == 1)}\n")

    # ── 1b. Hold out 20% as a final test set BEFORE any CV ──
    # This test set is never used during model selection or tuning.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42,
    )
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Held-out test set: {X_test.shape[0]} samples\n")

    # ── 2. Set up cross-validation ──
    # Stratified = each fold keeps the same healthy/pathological ratio
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── 3. Define three candidate algorithms ──
    # Each "pipeline" first standardises the data (zero mean, unit variance)
    # then feeds it to the classifier.  Standardising is important because
    # features like f0_mean (~178 Hz) are on a completely different scale
    # than jitter_local (~0.005), and some algorithms are sensitive to that.
    models = {
        # Random Forest: builds 100 decision trees, each voting independently.
        # max_depth=10 prevents individual trees from memorising the data.
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=100, max_depth=10,
                random_state=42, class_weight="balanced",
            )),
        ]),
        # SVM: finds a curved boundary between the two classes.
        # probability=True lets us get confidence scores, not just yes/no.
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf", probability=True,
                random_state=42, class_weight="balanced",
            )),
        ]),
        # Logistic Regression: the simplest model — a weighted sum of
        # features converted into a probability.
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000, random_state=42, class_weight="balanced",
            )),
        ]),
    }

    # ── 4. Evaluate all models side by side ──
    all_metrics = {}
    print("=" * 60)
    for name, pipeline in models.items():
        print(f"\n--- {name} ---")
        metrics = evaluate_model(pipeline, X_train, y_train, cv)
        all_metrics[name] = metrics
        for k, v in metrics.items():
            if not k.endswith("_std"):
                print(f"  {k}: {v:.3f} +/- {metrics[f'{k}_std']:.3f}")

    # ── 5. Pick the winner (highest F1 score) ──
    # F1 is chosen over plain accuracy because it balances precision (don't
    # cry wolf) and recall (don't miss sick people) — both matter in health.
    best_name = max(all_metrics, key=lambda n: all_metrics[n]["f1"])
    best_pipeline = models[best_name]
    print(f"\n{'=' * 60}")
    print(f"Best model (by F1): {best_name}")

    # ── 6. Build a confusion matrix for the best model ──
    # cross_val_predict gives us "out-of-fold" predictions: each sample is
    # predicted by a model that was NOT trained on it, so no cheating.
    y_pred = cross_val_predict(best_pipeline, X_train, y_train, cv=cv)
    cm = confusion_matrix(y_train, y_pred)
    report = classification_report(y_train, y_pred, target_names=["Healthy", "Pathological"])
    print(f"\nClassification Report ({best_name}) [CV on train set]:\n{report}")

    # ── 7. Train the winner on ALL training data so it's as strong as possible ──
    best_pipeline.fit(X_train, y_train)

    # ── 7b. Evaluate on held-out test set ──
    y_test_pred = best_pipeline.predict(X_test)
    y_test_proba = best_pipeline.predict_proba(X_test)[:, 1]
    test_metrics = {
        "accuracy": accuracy_score(y_test, y_test_pred),
        "precision": precision_score(y_test, y_test_pred),
        "recall": recall_score(y_test, y_test_pred),
        "f1": f1_score(y_test, y_test_pred),
        "roc_auc": roc_auc_score(y_test, y_test_proba),
    }
    print(f"\nHeld-out Test Set Results ({best_name}):")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.3f}")

    test_cm = confusion_matrix(y_test, y_test_pred)
    test_report = classification_report(y_test, y_test_pred, target_names=["Healthy", "Pathological"])
    print(f"\nTest Set Classification Report:\n{test_report}")

    # ── 7c. Refit on ALL data for deployment ──
    # (Cross-validation was just for evaluation — now we use everything.)
    best_pipeline.fit(X, y)

    # ── 8. ROC curve data (cross-validated probabilities on training set) ──
    y_proba = cross_val_predict(best_pipeline, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    fpr, tpr, _ = roc_curve(y_train, y_proba)
    roc_auc = auc(fpr, tpr)

    # ── 9. Feature importance ──
    # We always use Random Forest for this because it natively tells us
    # "which features did the decision trees split on most often?"
    # SVM and Logistic Regression don't give importance scores as easily.
    rf_pipeline = models["Random Forest"]
    rf_pipeline.fit(X, y)
    rf_importances = rf_pipeline.named_steps["clf"].feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf_importances,
    }).sort_values("importance", ascending=False)

    print(f"\nTop 10 Features (Random Forest):")
    print(importance_df.head(10).to_string(index=False))

    # ── 10. Save everything to model/ so the Streamlit app can load it ──
    os.makedirs("model", exist_ok=True)

    # The trained model itself (used by the app for real-time predictions)
    joblib.dump(best_pipeline, "model/rf_classifier.joblib")
    joblib.dump(feature_cols, "model/feature_names.joblib")

    save_metrics = {
        "best_model": best_name,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "cv_folds": 5,
        "data_source": data_source,
        "note": f"Trained on {data_source}. See load_voiced.py / generate_demo_data.py.",
        "models": {},
    }
    for name, m in all_metrics.items():
        save_metrics["models"][name] = {
            "accuracy": f"{m['accuracy']:.3f} +/- {m['accuracy_std']:.3f}",
            "precision": f"{m['precision']:.3f} +/- {m['precision_std']:.3f}",
            "recall": f"{m['recall']:.3f} +/- {m['recall_std']:.3f}",
            "f1_score": f"{m['f1']:.3f} +/- {m['f1_std']:.3f}",
            "auc_roc": f"{m['auc_roc']:.3f} +/- {m['auc_roc_std']:.3f}",
        }
    # Backward compat: top-level metrics for best model
    best_m = all_metrics[best_name]
    save_metrics["accuracy"] = f"{best_m['accuracy']:.3f} +/- {best_m['accuracy_std']:.3f}"
    save_metrics["precision"] = f"{best_m['precision']:.3f} +/- {best_m['precision_std']:.3f}"
    save_metrics["recall"] = f"{best_m['recall']:.3f} +/- {best_m['recall_std']:.3f}"
    save_metrics["f1_score"] = f"{best_m['f1']:.3f} +/- {best_m['f1_std']:.3f}"
    save_metrics["auc_roc"] = f"{best_m['auc_roc']:.3f} +/- {best_m['auc_roc_std']:.3f}"

    # Held-out test set metrics
    save_metrics["held_out_test"] = {
        "n_test": int(X_test.shape[0]),
        "accuracy": round(test_metrics["accuracy"], 4),
        "precision": round(test_metrics["precision"], 4),
        "recall": round(test_metrics["recall"], 4),
        "f1_score": round(test_metrics["f1"], 4),
        "auc_roc": round(test_metrics["roc_auc"], 4),
    }

    with open("model/metrics.json", "w") as f:
        json.dump(save_metrics, f, indent=2)

    cm_df = pd.DataFrame(cm,
        index=["Actual Healthy", "Actual Pathological"],
        columns=["Pred Healthy", "Pred Pathological"])
    cm_df.to_csv("model/confusion_matrix.csv")

    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    roc_df.to_csv("model/roc_data.csv", index=False)

    importance_df.to_csv("model/feature_importance.csv", index=False)

    # ── 11. Precision-Recall curve data ──
    prec, rec, pr_thresh = precision_recall_curve(y_train, y_proba)
    avg_prec = average_precision_score(y_train, y_proba)
    pr_df = pd.DataFrame({
        "precision": prec[:-1],
        "recall": rec[:-1],
        "threshold": pr_thresh,
    })
    pr_df.to_csv("model/pr_curve.csv", index=False)
    print(f"\nAverage Precision: {avg_prec:.3f}")

    # ── 12. Bootstrap confidence intervals ──
    # Resample predictions 1000 times to get 95% CI on each metric
    y_pred_cv = (y_proba >= 0.5).astype(int)
    n_boot = 1000
    rng = np.random.RandomState(42)
    boot_results = {m: [] for m in ["accuracy", "f1", "recall", "precision", "roc_auc"]}
    for _ in range(n_boot):
        idx = rng.choice(len(y_train), len(y_train), replace=True)
        boot_results["accuracy"].append(accuracy_score(y_train[idx], y_pred_cv[idx]))
        boot_results["f1"].append(f1_score(y_train[idx], y_pred_cv[idx]))
        boot_results["recall"].append(recall_score(y_train[idx], y_pred_cv[idx]))
        boot_results["precision"].append(precision_score(y_train[idx], y_pred_cv[idx]))
        boot_results["roc_auc"].append(roc_auc_score(y_train[idx], y_proba[idx]))

    ci_data = {}
    print(f"\n95% Bootstrap Confidence Intervals ({n_boot} iterations):")
    for m, values in boot_results.items():
        low, high = np.percentile(values, [2.5, 97.5])
        mean = np.mean(values)
        ci_data[m] = {"mean": round(mean, 4), "ci_low": round(low, 4), "ci_high": round(high, 4)}
        print(f"  {m:12s}: {mean:.3f}  [{low:.3f}, {high:.3f}]")

    save_metrics["bootstrap_ci"] = ci_data
    save_metrics["average_precision"] = round(avg_prec, 4)

    # Re-save metrics.json with new fields
    with open("model/metrics.json", "w") as f:
        json.dump(save_metrics, f, indent=2)

    # ── 13. Learning curves ──
    # Shows how performance changes with data size
    print("\nComputing learning curves...")
    train_sizes_abs, train_scores, val_scores = learning_curve(
        best_pipeline, X_train, y_train, cv=cv,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="f1", n_jobs=-1, random_state=42,
    )
    lc_df = pd.DataFrame({
        "train_size": train_sizes_abs,
        "train_mean": train_scores.mean(axis=1),
        "train_std": train_scores.std(axis=1),
        "val_mean": val_scores.mean(axis=1),
        "val_std": val_scores.std(axis=1),
    })
    lc_df.to_csv("model/learning_curve.csv", index=False)
    print(f"Learning curve saved (final val F1: {val_scores.mean(axis=1)[-1]:.3f})")

    print(f"\nSaved all artifacts to model/")


if __name__ == "__main__":
    main()

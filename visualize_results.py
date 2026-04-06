"""
Generate publication-quality visualizations from trained model artifacts.

Produces radar charts, feature importance plots, ROC/PR curves, and
confusion matrix heatmaps from the saved model outputs.

Usage:
    python visualize_results.py          # generate all plots
    python visualize_results.py --show   # also display interactively

Outputs saved to screenshots/:
    radar_healthy_vs_pathological.png
    feature_importance_top15.png
    model_comparison.png
    confusion_matrix.png
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend by default


def plot_radar_chart(save_dir):
    """Radar chart comparing healthy vs pathological feature profiles."""
    real_path = "data/real_features.csv"
    synth_path = "data/features.csv"
    data_path = real_path if os.path.exists(real_path) else synth_path
    df = pd.read_csv(data_path)

    healthy = df[df["label"] == 0]
    pathological = df[df["label"] == 1]

    features = ["jitter_local", "shimmer_local", "hnr", "f0_std",
                "spectral_flatness", "zcr", "rms", "mfcc_1"]
    labels = ["Jitter", "Shimmer", "HNR", "F0 Std",
              "Spec. Flatness", "ZCR", "RMS", "MFCC-1"]

    # Normalize each feature to [0, 1] using min-max across both groups
    healthy_vals = []
    pathological_vals = []
    for feat in features:
        all_vals = df[feat]
        fmin, fmax = all_vals.min(), all_vals.max()
        h_norm = (healthy[feat].mean() - fmin) / (fmax - fmin) if fmax > fmin else 0.5
        p_norm = (pathological[feat].mean() - fmin) / (fmax - fmin) if fmax > fmin else 0.5
        healthy_vals.append(h_norm)
        pathological_vals.append(p_norm)

    # Close the polygon
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]
    healthy_vals += healthy_vals[:1]
    pathological_vals += pathological_vals[:1]
    labels_closed = labels + [labels[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, healthy_vals, alpha=0.2, color="#2e7d32", label="Healthy")
    ax.plot(angles, healthy_vals, "o-", color="#2e7d32", linewidth=2)
    ax.fill(angles, pathological_vals, alpha=0.2, color="#d32f2f", label="Pathological")
    ax.plot(angles, pathological_vals, "o-", color="#d32f2f", linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_title("Feature Profile: Healthy vs Pathological", fontsize=14,
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.1), fontsize=11)
    ax.set_ylim(0, 1)

    path = os.path.join(save_dir, "radar_healthy_vs_pathological.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_feature_importance(save_dir):
    """Horizontal bar chart of top 15 feature importances."""
    importance = pd.read_csv("model/feature_importance.csv")
    top = importance.head(15).copy()

    friendly = {
        "f0_mean": "Base Pitch", "f0_std": "Pitch Wobble",
        "jitter_local": "Jitter", "jitter_rap": "Jitter RAP",
        "shimmer_local": "Shimmer", "shimmer_apq3": "Shimmer APQ3",
        "hnr": "Voice Clarity (HNR)",
        "spectral_centroid": "Brightness", "spectral_bandwidth": "Freq. Spread",
        "spectral_flatness": "Noisiness", "spectral_rolloff": "High-Freq Energy",
        "zcr": "Zero-Crossing Rate", "rms": "Loudness",
    }
    for i in range(1, 14):
        friendly[f"mfcc_{i}"] = f"MFCC-{i}"
    top["feature"] = top["feature"].map(lambda x: friendly.get(x, x))

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top)))[::-1]
    ax.barh(range(len(top)), top["importance"].values[::-1], color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values[::-1], fontsize=10)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title("Top 15 Feature Importances (Random Forest)", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    path = os.path.join(save_dir, "feature_importance_top15.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_model_comparison(save_dir):
    """Grouped bar chart comparing all three models."""
    with open("model/metrics.json") as f:
        metrics = json.load(f)

    if "models" not in metrics:
        print("  Skipped model comparison (no multi-model data)")
        return

    model_names = list(metrics["models"].keys())
    metric_keys = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"]

    # Parse "0.729 +/- 0.092" format
    data = {}
    for name in model_names:
        data[name] = []
        for key in metric_keys:
            val_str = metrics["models"][name].get(key, "0")
            val = float(val_str.split(" +/- ")[0])
            data[name].append(val)

    x = np.arange(len(metric_labels))
    width = 0.25
    colors = ["#2563eb", "#d32f2f", "#2e7d32"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, name in enumerate(model_names):
        ax.bar(x + i * width, data[name], width, label=name, color=colors[i], alpha=0.85)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison (5-Fold CV)", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(save_dir, "model_comparison.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrix(save_dir):
    """Publication-quality confusion matrix heatmap."""
    cm = pd.read_csv("model/confusion_matrix.csv", index_col=0)
    labels = ["Healthy", "Pathological"]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm.values, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("Actual", fontsize=13)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")

    for i in range(2):
        for j in range(2):
            color = "white" if cm.values[i, j] > cm.values.max() / 2 else "black"
            ax.text(j, i, str(cm.values[i, j]), ha="center", va="center",
                    fontsize=20, fontweight="bold", color=color)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate VocalPath visualizations")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    args = parser.parse_args()

    if args.show:
        matplotlib.use("TkAgg")

    save_dir = "screenshots"
    os.makedirs(save_dir, exist_ok=True)

    print("Generating visualizations...")
    plot_radar_chart(save_dir)
    plot_feature_importance(save_dir)
    plot_model_comparison(save_dir)
    plot_confusion_matrix(save_dir)
    print(f"\nAll visualizations saved to {save_dir}/")


if __name__ == "__main__":
    main()

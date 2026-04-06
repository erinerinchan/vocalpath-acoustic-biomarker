"""Generate a demo image for the README showing key VocalPath outputs."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import json
import os

np.random.seed(42)

# --- Load real project data ---
df = pd.read_csv("data/features.csv")
with open("model/metrics.json") as f:
    metrics = json.load(f)
roc = pd.read_csv("model/roc_data.csv")
importance = pd.read_csv("model/feature_importance.csv").sort_values("importance", ascending=False).head(10)

# Friendly feature names
friendly = {
    "jitter_local": "Jitter (pitch wobble)",
    "shimmer_local": "Shimmer (volume wobble)",
    "hnr": "Harmonic-to-Noise Ratio",
    "f0_mean": "Avg Pitch (F0)",
    "f0_std": "Pitch Variability",
    "mfcc_1": "Vocal Tone Shape 1",
    "mfcc_2": "Vocal Tone Shape 2",
    "mfcc_3": "Vocal Tone Shape 3",
    "spectral_centroid": "Brightness",
    "spectral_flatness": "Noise vs Tone",
    "spectral_bandwidth": "Frequency Spread",
    "spectral_rolloff": "High-Freq Energy Cutoff",
    "zcr": "Zero Crossing Rate",
    "rms": "Volume (RMS)",
}

# --- Create figure ---
fig = plt.figure(figsize=(14, 7), facecolor="#f8f9fa")
fig.suptitle("VocalPath: Acoustic Biomarker Analysis", fontsize=18, fontweight="bold",
             color="#1e3a8a", y=0.97)

# 1. Simulated waveform (top-left)
ax1 = fig.add_subplot(2, 3, 1)
t = np.linspace(0, 1, 16000)
wave = np.sin(2 * np.pi * 150 * t) * np.exp(-0.5 * t) * (1 + 0.3 * np.sin(2 * np.pi * 5 * t))
wave += 0.05 * np.random.randn(len(t))
ax1.plot(t, wave, color="#2563eb", linewidth=0.3)
ax1.set_title("Waveform", fontsize=10, fontweight="bold", color="#1e3a8a")
ax1.set_xlabel("Time (s)", fontsize=8)
ax1.set_ylabel("Amplitude", fontsize=8)
ax1.tick_params(labelsize=7)

# 2. Radar chart (top-center)
ax2 = fig.add_subplot(2, 3, 2, polar=True)
labels = ["Jitter", "Shimmer", "HNR", "F0", "MFCC-1", "Brightness"]
healthy = [0.3, 0.25, 0.8, 0.6, 0.5, 0.4]
pathological = [0.7, 0.65, 0.35, 0.45, 0.7, 0.6]
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
healthy += healthy[:1]
pathological += pathological[:1]
angles += angles[:1]
ax2.plot(angles, healthy, "o-", color="#22c55e", linewidth=1.5, markersize=4, label="Healthy")
ax2.fill(angles, healthy, alpha=0.15, color="#22c55e")
ax2.plot(angles, pathological, "o-", color="#ef4444", linewidth=1.5, markersize=4, label="Pathological")
ax2.fill(angles, pathological, alpha=0.15, color="#ef4444")
ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(labels, fontsize=7)
ax2.set_title("Biomarker Radar", fontsize=10, fontweight="bold", color="#1e3a8a", pad=15)
ax2.legend(fontsize=6, loc="upper right", bbox_to_anchor=(1.3, 1.1))

# 3. ROC Curve (top-right)
ax3 = fig.add_subplot(2, 3, 3)
ax3.plot(roc["fpr"], roc["tpr"], color="#2563eb", linewidth=2,
         label=f"AUC = {metrics['auc_roc'].split()[0]}")
ax3.plot([0, 1], [0, 1], "--", color="#94a3b8", linewidth=1)
ax3.fill_between(roc["fpr"], roc["tpr"], alpha=0.1, color="#2563eb")
ax3.set_title("ROC Curve", fontsize=10, fontweight="bold", color="#1e3a8a")
ax3.set_xlabel("False Positive Rate", fontsize=8)
ax3.set_ylabel("True Positive Rate", fontsize=8)
ax3.legend(fontsize=8)
ax3.tick_params(labelsize=7)

# 4. Feature importance (bottom-left)
ax4 = fig.add_subplot(2, 3, 4)
feat_names = [friendly.get(f, f) for f in importance["feature"]]
colors = ["#2563eb" if i < 3 else "#60a5fa" for i in range(len(feat_names))]
ax4.barh(range(len(feat_names)), importance["importance"].values, color=colors)
ax4.set_yticks(range(len(feat_names)))
ax4.set_yticklabels(feat_names, fontsize=7)
ax4.invert_yaxis()
ax4.set_title("Top Features", fontsize=10, fontweight="bold", color="#1e3a8a")
ax4.set_xlabel("Importance", fontsize=8)
ax4.tick_params(labelsize=7)

# 5. Confusion matrix (bottom-center)
ax5 = fig.add_subplot(2, 3, 5)
cm = pd.read_csv("model/confusion_matrix.csv", index_col=0).values
im = ax5.imshow(cm, cmap="Blues", aspect="auto")
for i in range(2):
    for j in range(2):
        ax5.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center",
                 fontsize=14, fontweight="bold",
                 color="white" if cm[i, j] > cm.max() / 2 else "#1e3a8a")
ax5.set_xticks([0, 1])
ax5.set_yticks([0, 1])
ax5.set_xticklabels(["Healthy", "Pathological"], fontsize=8)
ax5.set_yticklabels(["Healthy", "Pathological"], fontsize=8)
ax5.set_title("Confusion Matrix", fontsize=10, fontweight="bold", color="#1e3a8a")
ax5.set_xlabel("Predicted", fontsize=8)
ax5.set_ylabel("Actual", fontsize=8)

# 6. Class distribution (bottom-right)
ax6 = fig.add_subplot(2, 3, 6)
counts = df["label"].value_counts().sort_index()
bars = ax6.bar(["Healthy", "Pathological"], counts.values,
               color=["#22c55e", "#ef4444"], edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, counts.values):
    ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
             str(val), ha="center", fontsize=10, fontweight="bold", color="#1e3a8a")
ax6.set_title("Dataset Balance", fontsize=10, fontweight="bold", color="#1e3a8a")
ax6.set_ylabel("Samples", fontsize=8)
ax6.tick_params(labelsize=7)

plt.tight_layout(rect=[0, 0, 1, 0.93])

os.makedirs("screenshots", exist_ok=True)
fig.savefig("screenshots/demo.png", dpi=150, bbox_inches="tight", facecolor="#f8f9fa")
plt.close()
print("Saved screenshots/demo.png")

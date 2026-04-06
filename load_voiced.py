"""
Load the VOICED clinical dataset and extract 26 acoustic features.

The VOICED (VOice ICar fEDerico II) database contains 208 voice recordings
from the University of Naples Federico II hospital:
  - 58 healthy speakers
  - 150 speakers with clinically verified voice pathologies
  - Diagnoses confirmed via the SIFEL protocol (Italian Society of
    Phoniatrics and Logopaedics)

Each recording is a 5-second sustained vowel 'a' at 8 kHz / 32-bit.

Reference:
    Cesari, U. et al. (2018). "A new database of healthy and pathological
    voices." Computers & Electrical Engineering, 68, 310-321.

Usage:
    python load_voiced.py

Output:
    data/real_features.csv  — 208 rows x 27 columns (26 features + label)
"""

import os
import re
import numpy as np
import pandas as pd
import parselmouth
from feature_extraction import extract_features_from_audio, FEATURE_NAMES

# ── Configuration ──
VOICED_DIR = os.path.join("data", "voiced", "VOICED DATASET")
OUTPUT_PATH = os.path.join("data", "real_features.csv")
SAMPLE_RATE = 8000  # VOICED dataset sample rate


def parse_diagnosis(info_path):
    """Read the -info.txt file and return (diagnosis_str, age, sex)."""
    diagnosis = None
    age = None
    sex = None
    with open(info_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("Diagnosis:"):
                diagnosis = line.split(":", 1)[1].strip().lower()
            elif line.startswith("Age:"):
                try:
                    age = int(line.split(":", 1)[1].strip())
                except ValueError:
                    age = None
            elif line.startswith("Gender:"):
                sex = line.split(":", 1)[1].strip().lower()
    return diagnosis, age, sex


def load_signal_from_txt(txt_path):
    """Load the voice signal from the plain-text .txt file."""
    samples = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(float(line))
    return np.array(samples, dtype=np.float64)


def is_healthy(diagnosis):
    """Map VOICED diagnosis string to binary label (0=healthy, 1=pathological)."""
    return diagnosis is not None and diagnosis.strip() == "healthy"


def main():
    print("Loading VOICED dataset from:", VOICED_DIR)
    print("=" * 60)

    # Find all voice IDs
    pattern = re.compile(r"^voice(\d+)\.txt$")
    voice_ids = []
    for fname in sorted(os.listdir(VOICED_DIR)):
        m = pattern.match(fname)
        if m:
            voice_ids.append(int(m.group(1)))

    print(f"Found {len(voice_ids)} voice recordings\n")

    rows = []
    skipped = 0

    for vid in voice_ids:
        prefix = f"voice{vid:03d}"
        txt_path = os.path.join(VOICED_DIR, f"{prefix}.txt")
        info_path = os.path.join(VOICED_DIR, f"{prefix}-info.txt")

        if not os.path.exists(txt_path) or not os.path.exists(info_path):
            print(f"  SKIP {prefix}: missing files")
            skipped += 1
            continue

        diagnosis, age, sex = parse_diagnosis(info_path)
        label = 0 if is_healthy(diagnosis) else 1

        try:
            # Load signal from text file
            signal = load_signal_from_txt(txt_path)

            if len(signal) < SAMPLE_RATE:  # less than 1 second
                print(f"  SKIP {prefix}: too short ({len(signal)} samples)")
                skipped += 1
                continue

            # Normalize signal
            max_val = np.max(np.abs(signal))
            if max_val > 0:
                signal = signal / max_val

            # Create Praat Sound object from numpy array
            sound = parselmouth.Sound(signal, sampling_frequency=SAMPLE_RATE)

            # For librosa features, use the signal directly
            y = signal.astype(np.float32)
            sr = SAMPLE_RATE

            features = extract_features_from_audio(sound, y, sr)

            # Replace any NaN/undefined Praat values with 0
            for k, v in features.items():
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    features[k] = 0.0

            features["label"] = label
            features["source_id"] = prefix
            features["diagnosis"] = diagnosis
            features["age"] = age
            features["sex"] = sex
            rows.append(features)

            status = "healthy" if label == 0 else "pathological"
            print(f"  {prefix}: {status} ({diagnosis}) — {len(signal)/SAMPLE_RATE:.1f}s")

        except Exception as e:
            print(f"  SKIP {prefix}: {e}")
            skipped += 1

    print(f"\n{'=' * 60}")
    print(f"Processed: {len(rows)}")
    print(f"Skipped:   {skipped}")

    if not rows:
        print("ERROR: No samples processed. Check VOICED_DIR path.")
        return

    df = pd.DataFrame(rows)

    # Reorder columns: features first, then metadata
    meta_cols = ["source_id", "diagnosis", "age", "sex"]
    feature_order = FEATURE_NAMES + ["label"] + meta_cols
    df = df[feature_order]

    # Save features-only version (matching synthetic data format)
    df_features = df[FEATURE_NAMES + ["label"]]
    df_features.to_csv(OUTPUT_PATH, index=False)

    # Also save full version with metadata
    df.to_csv(OUTPUT_PATH.replace(".csv", "_full.csv"), index=False)

    print(f"\nSaved {len(df)} samples to {OUTPUT_PATH}")
    print(f"  Healthy:      {(df['label'] == 0).sum()}")
    print(f"  Pathological: {(df['label'] == 1).sum()}")

    # Summary statistics
    print(f"\nFeature summary (real data):")
    for feat in ["jitter_local", "shimmer_local", "hnr", "f0_mean"]:
        h = df_features[df_features["label"] == 0][feat]
        p = df_features[df_features["label"] == 1][feat]
        print(f"  {feat:20s}  healthy={h.mean():.4f} +/- {h.std():.4f}"
              f"  pathological={p.mean():.4f} +/- {p.std():.4f}")


if __name__ == "__main__":
    main()

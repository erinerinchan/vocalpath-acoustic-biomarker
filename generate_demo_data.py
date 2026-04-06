"""
Generate synthetic voice feature data with realistic class overlap.

Why synthetic data?
    We don't have access to real patient recordings, so we create fake voice
    measurements that *behave* like real clinical data.  Each fake patient gets
    a hidden "how-sick-are-they" score, and all 26 voice measurements shift
    together based on that score — just like in real life, where someone with
    vocal cord damage tends to have multiple abnormal readings at once.

    The healthy and pathological ranges deliberately overlap, because in
    reality many borderline patients look "almost healthy" on paper.

Data is modeled on published clinical ranges from:
- Teixeira et al. (2013) "Vocal Acoustic Analysis - Jitter, Shimmer and HNR Parameters"
  Procedia Technology, 9, 1112-1122.
- Godino-Llorente et al. (2006) "Dimensionality Reduction of a Pathological Voice Quality
  Feature Set" IEEE Trans. Biomedical Engineering, 53(10).
- Martinez et al. (2012) "Voice Pathology Detection on the Saarbrucken Voice Database"

NOTE: This data is SYNTHETIC for demonstration. For production use, replace with
real clinical datasets (e.g., SVD, AVFAD) via feature_extraction.py.
"""

import numpy as np
import pandas as pd
import os

# ── Feature definitions ──
# Each feature has four numbers:
#   (healthy_average, healthy_spread, pathological_average, pathological_spread)
#
# "average" = the typical value for that group
# "spread"  = how much individual patients vary around that average
#             (technically the standard deviation of a normal distribution)
#
# When the healthy and pathological ranges overlap a lot (e.g., f0_mean),
# the feature is a weak signal on its own.  When they're more separated
# (e.g., jitter_local), the feature is a stronger signal.
FEATURE_PARAMS = {
    # --- Voice stability measures (strongest signal) ---
    # f0 = base pitch in Hz; doesn't change much between groups
    "f0_mean":           (178, 65,   168, 72),
    # f0_std = how much the pitch wobbles; pathological voices wobble more
    "f0_std":            (3.8, 2.2,  6.0, 3.5),
    # jitter = pitch instability from one vocal-cord vibration to the next
    "jitter_local":      (0.005, 0.003,  0.010, 0.006),
    "jitter_rap":        (0.003, 0.002,  0.006, 0.004),
    # shimmer = volume instability from one vibration to the next
    "shimmer_local":     (0.028, 0.016,  0.052, 0.028),
    "shimmer_apq3":      (0.014, 0.008,  0.028, 0.014),
    # HNR = voice clarity; healthy voices score higher (more tone, less noise)
    "hnr":               (21.5, 5.0,  16.5, 5.5),

    # --- MFCCs (vocal tract shape — moderate signal) ---
    # Think of these as a 13-number "fingerprint" of the mouth/throat shape.
    # Each one captures a different frequency band of the voice.
    "mfcc_1":            (-210, 55,  -240, 62),
    "mfcc_2":            (48, 32,    38, 36),
    "mfcc_3":            (-10, 22,   -14, 25),
    "mfcc_4":            (19, 16,    15, 18),
    "mfcc_5":            (-5, 13,    -7, 15),
    "mfcc_6":            (4, 11,     2, 12),
    "mfcc_7":            (-8, 11,    -10, 12),
    "mfcc_8":            (3, 9,      1, 10),
    "mfcc_9":            (-2, 9,     -4, 10),
    "mfcc_10":           (1, 8,      -1, 9),
    "mfcc_11":           (-3, 8,     -5, 8),
    "mfcc_12":           (2, 7,      0, 7),
    "mfcc_13":           (-1, 7,     -3, 7),

    # --- Spectral & energy features (weakest signal individually) ---
    # These describe the overall "texture" and loudness of the voice.
    "spectral_centroid":  (1550, 450,  1750, 500),   # brightness
    "spectral_bandwidth": (1850, 350,  2050, 400),   # frequency spread
    "spectral_flatness":  (0.025, 0.015,  0.050, 0.028),  # noisy vs tonal
    "spectral_rolloff":   (3100, 850,  3400, 950),   # high-frequency energy
    "zcr":               (0.052, 0.025,  0.072, 0.030),  # zero-crossing rate
    "rms":               (0.078, 0.032,  0.062, 0.032),  # loudness (lower in sick voices)
}

# Features that get WORSE (increase) as vocal cord damage increases
POSITIVE_SEVERITY = [
    "jitter_local", "jitter_rap", "shimmer_local", "shimmer_apq3",
    "f0_std", "spectral_flatness", "zcr", "spectral_centroid",
    "spectral_bandwidth", "spectral_rolloff",
]
# Features that get LOWER as vocal cord damage increases
NEGATIVE_SEVERITY = ["hnr", "rms"]


def generate_correlated_samples(n_samples, label, rng):
    """
    Create fake patients whose voice measurements are realistically linked.

    The key trick: each patient gets a random "severity" score.  If a patient
    happens to draw a high severity, ALL their measurements shift in the
    unhealthy direction together — jitter goes up, shimmer goes up, HNR goes
    down, etc.  This mimics real life, where a person with bad vocal cord
    damage doesn't just have one abnormal reading; they tend to have many.

    Without this, every measurement would be randomly independent, and the
    data would be too "clean" compared to real clinical recordings.
    """
    features = list(FEATURE_PARAMS.keys())
    data = []

    for _ in range(n_samples):
        # Hidden severity score — a random nudge that pushes ALL features
        # in the same direction for this patient.  0.3 controls how strongly
        # linked the features are (higher = more correlated).
        severity = rng.normal(0, 0.3)

        sample = {}
        for feat in features:
            h_mu, h_std, p_mu, p_std = FEATURE_PARAMS[feat]
            mu = p_mu if label == 1 else h_mu   # pick healthy or sick average
            std = p_std if label == 1 else h_std  # pick healthy or sick spread

            # Start with a random value around the group average
            base_val = rng.normal(mu, std)

            # Then nudge it based on the patient's severity score.
            # 0.4 = nudge strength; keeps the shift moderate (not overwhelming).
            if feat in POSITIVE_SEVERITY:
                base_val += severity * std * 0.4   # sicker → higher
            elif feat in NEGATIVE_SEVERITY:
                base_val -= severity * std * 0.4   # sicker → lower

            sample[feat] = base_val

        sample["label"] = label
        data.append(sample)

    return data


def add_label_noise(df, noise_rate=0.05, rng=None):
    """
    Randomly flip 5% of labels (healthy ↔ pathological).

    Why? In real clinical practice, some patients are mis-diagnosed or sit
    right on the borderline.  Adding this noise makes the training data
    more realistic and prevents the model from memorising the data too
    perfectly (which would inflate accuracy and not generalise well).
    """
    n_flip = int(len(df) * noise_rate)
    flip_idx = rng.choice(df.index, size=n_flip, replace=False)
    df.loc[flip_idx, "label"] = 1 - df.loc[flip_idx, "label"]
    return df, n_flip


if __name__ == "__main__":
    rng = np.random.default_rng(42)  # fixed seed so results are reproducible

    # Create 300 fake healthy patients and 300 fake pathological patients
    healthy = generate_correlated_samples(300, 0, rng)
    pathological = generate_correlated_samples(300, 1, rng)

    df = pd.DataFrame(healthy + pathological)

    # Flip 5% of labels to simulate real-world diagnostic uncertainty
    df, n_flipped = add_label_noise(df, noise_rate=0.05, rng=rng)

    # Some random values might end up physically impossible (e.g., negative
    # jitter).  Clamp them to sensible minimums so the data stays realistic.
    clamp_cols = {
        "jitter_local": 0.0001, "jitter_rap": 0.0001,
        "shimmer_local": 0.001, "shimmer_apq3": 0.001,
        "hnr": 0, "spectral_flatness": 0.001,
        "zcr": 0.001, "rms": 0.001,
        "f0_mean": 50, "f0_std": 0.1,
    }
    for col, lower in clamp_cols.items():
        df[col] = df[col].clip(lower=lower)

    # Shuffle so healthy and pathological rows aren't grouped together
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/features.csv", index=False)
    print(f"Generated {len(df)} samples -> data/features.csv")
    print(f"  Healthy: {sum(df['label'] == 0)}, Pathological: {sum(df['label'] == 1)}")
    print(f"  Features: {len(df.columns) - 1}")
    print(f"  Label noise: {n_flipped} samples flipped ({n_flipped/len(df)*100:.1f}%)")
    print(f"  Inter-feature correlations: enabled (latent severity model)")

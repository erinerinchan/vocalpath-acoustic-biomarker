"""
Train a mel-spectrogram CNN for voice pathology classification.

What this script does (plain English):
    The main pipeline (train_model.py) extracts 26 numbers from each voice
    recording and feeds them into a traditional ML model.  This script takes
    a fundamentally different approach:

    1. Generates synthetic voice audio clips (healthy + pathological)
    2. Converts each clip into a **mel-spectrogram** — a 2D image showing
       how the voice's frequency content changes over time
    3. Trains a **Convolutional Neural Network (CNN)** to classify these
       spectrogram images directly

    Why does this matter?
    ─────────────────────
    Traditional models flatten each recording into a single row of numbers,
    losing the *time dimension*.  A CNN on spectrograms preserves it — it
    can learn patterns like "the pitch wobbles more in the second half" or
    "high-frequency noise builds up over time."

    This is the same family of techniques used in:
      • Modern speech recognition (e.g., Google Voice, Siri)
      • Speaker identification / voice biometrics
      • Music genre classification
      • Environmental sound detection

    The main app still uses the tabular pipeline because it is more
    interpretable for clinicians.  This script demonstrates deeper
    audio-ML and deep-learning capability.

Requirements (install only if you want to run this script):
    pip install tensorflow librosa numpy scikit-learn

Outputs:
    model/spectrogram_cnn.keras   – the trained CNN model
    model/cnn_metrics.json        – accuracy, AUC, and training details
"""

import numpy as np
import librosa
import os
import json


# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — SYNTHESIZE VOICE AUDIO
# ═══════════════════════════════════════════════════════════════════════════
# We reuse the same synthesis logic from generate_sample_audio.py, but with
# more variation so the CNN has enough diversity to learn from.

def synthesize_vowel(duration=3.0, sr=22050, f0=180, noise_level=0.02,
                     shimmer_amount=0.0, jitter_amount=0.0,
                     sub_harmonic=0.0, noise_ramp=False):
    """
    Create a synthetic sustained vowel ('ah') with controllable pathology
    markers.

    Parameters (plain English):
        duration       – length of the clip in seconds
        sr             – sample rate (samples per second; 22 050 is standard)
        f0             – base pitch in Hz (higher = higher voice)
        noise_level    – how breathy / noisy the voice is (0 = pure tone)
        shimmer_amount – random volume wobble (0 = perfectly steady volume)
        jitter_amount  – cycle-to-cycle pitch variation in Hz (0 = steady)
        sub_harmonic   – amplitude of a half-frequency component (diplophonia)
                         — in real pathology, damaged vocal cords sometimes
                         vibrate at half their normal rate, creating a
                         characteristic low "rumble" in the spectrogram
        noise_ramp     – if True, noise increases over time (simulates vocal
                         fatigue — the voice gets breathier as the patient
                         sustains the vowel)
    """
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    # Jitter: random pitch wobble applied per-cycle
    # We modulate the instantaneous frequency so each vocal-cord cycle
    # is slightly different from the last — visible as blurred harmonics
    # in the spectrogram
    if jitter_amount > 0:
        # Create per-cycle frequency variation (not divided by sr)
        inst_freq = f0 + jitter_amount * np.random.randn(n_samples)
        phase = 2 * np.pi * np.cumsum(inst_freq) / sr
    else:
        phase = 2 * np.pi * f0 * t

    # Build the voice signal from a fundamental frequency + harmonics
    # (this is how real vocal cords work — they produce a "buzz" made
    # of the base pitch layered with multiples of that pitch)
    signal = np.zeros(n_samples, dtype=np.float64)
    harmonics = [1.0, 0.5, 0.3, 0.2, 0.1, 0.07, 0.04]
    for i, amp in enumerate(harmonics, 1):
        signal += amp * np.sin(i * phase)

    # Sub-harmonic: a component at half the fundamental frequency
    # This creates a distinctive pattern in the spectrogram that only
    # appears in pathological voices (called diplophonia)
    if sub_harmonic > 0:
        signal += sub_harmonic * np.sin(0.5 * phase)

    # Formant resonances for an /a/ vowel — these simulate the filtering
    # effect of the mouth and throat cavity
    for freq, amp in [(730, 0.3), (1090, 0.15), (2440, 0.08)]:
        signal += amp * np.sin(2 * np.pi * freq * t)

    # Breathiness: additive noise (like whispering mixed into the voice)
    if noise_ramp:
        # Noise increases over time — simulates vocal fatigue
        ramp = np.linspace(0.3, 1.0, n_samples)
        signal += noise_level * ramp * np.random.randn(n_samples)
    else:
        signal += noise_level * np.random.randn(n_samples)

    # Shimmer: random amplitude modulation (volume jumping up and down)
    if shimmer_amount > 0:
        signal *= (1.0 + shimmer_amount * np.random.randn(n_samples))

    # Gentle fade-in / fade-out so the clip doesn't start/end abruptly
    envelope = np.ones(n_samples)
    attack = int(0.1 * sr)
    release = int(0.15 * sr)
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-release:] = np.linspace(1, 0, release)
    signal *= envelope

    # Normalize to a safe amplitude
    signal = signal / (np.max(np.abs(signal)) + 1e-8) * 0.8
    return signal.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — GENERATE DATASET OF MEL-SPECTROGRAMS
# ═══════════════════════════════════════════════════════════════════════════
# A mel-spectrogram is a 2D array where:
#   • Each column = a short time window
#   • Each row    = a frequency band (on the mel scale, which mimics
#                   how the human ear perceives pitch)
#   • The value   = how loud that frequency band is in that time window
#
# So a single spectrogram is like a heatmap photo of the voice.
# A CNN can look at this "photo" and learn visual patterns that
# distinguish healthy from pathological voices.

def generate_dataset(n_per_class=200, sr=22050, duration=3.0):
    """
    Generate synthetic audio samples and compute mel-spectrograms.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 128, time_steps)
        Mel-spectrograms in decibels.
    y : np.ndarray, shape (n_samples,)
        Labels: 0 = healthy, 1 = pathological.
    """
    spectrograms = []
    labels = []

    print(f"  Generating {n_per_class} healthy samples...")
    for _ in range(n_per_class):
        # Healthy voices: normal pitch, low noise, no shimmer/jitter
        f0 = np.random.normal(178, 20)
        noise = np.random.uniform(0.005, 0.02)
        audio = synthesize_vowel(duration=duration, sr=sr, f0=f0,
                                 noise_level=noise)
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128,
                                             fmax=8000)
        # Use ref=1.0 (fixed) so absolute power levels are preserved.
        # ref=np.max would normalize each sample to its own loudest point,
        # erasing the very amplitude differences we want the CNN to learn.
        mel_db = librosa.power_to_db(mel, ref=1.0)
        spectrograms.append(mel_db)
        labels.append(0)

    print(f"  Generating {n_per_class} pathological samples...")
    for _ in range(n_per_class):
        # Pathological voices: more noise, shimmer, jitter, sub-harmonics
        # These mimic the acoustic effects of vocal cord lesions, nodules,
        # or early laryngeal changes that disrupt normal vibration
        f0 = np.random.normal(165, 25)
        noise = np.random.uniform(0.15, 0.35)       # much breathier
        shimmer = np.random.uniform(0.08, 0.20)      # strong volume wobble
        jitter = np.random.uniform(3.0, 8.0)         # noticeable pitch instability (Hz)
        sub_harm = np.random.uniform(0.1, 0.3)       # diplophonia (half-freq component)
        use_ramp = np.random.random() > 0.5           # 50% chance of vocal fatigue
        audio = synthesize_vowel(duration=duration, sr=sr, f0=f0,
                                 noise_level=noise, shimmer_amount=shimmer,
                                 jitter_amount=jitter, sub_harmonic=sub_harm,
                                 noise_ramp=use_ramp)
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128,
                                             fmax=8000)
        mel_db = librosa.power_to_db(mel, ref=1.0)
        spectrograms.append(mel_db)
        labels.append(1)

    return np.array(spectrograms), np.array(labels)


# ═══════════════════════════════════════════════════════════════════════════
# PART 3 — BUILD AND TRAIN THE CNN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # --- Step 1: Generate data ---
    print("=" * 55)
    print("SPECTROGRAM CNN — Voice Pathology Classification")
    print("=" * 55)
    print("\nStep 1/4: Generating synthetic audio dataset...")
    X, y = generate_dataset(n_per_class=400)

    # CNN expects shape (samples, height, width, channels)
    # Our spectrogram is (samples, 128, time_steps) — we add a
    # channel dimension, like a grayscale image
    X = X[..., np.newaxis]

    # Per-sample standardisation (mean=0, std=1 for each spectrogram).
    # This preserves the *shape* differences between healthy and
    # pathological spectrograms while putting all samples on a common scale.
    for i in range(len(X)):
        m, s = X[i].mean(), X[i].std()
        X[i] = (X[i] - m) / (s + 1e-8)

    print(f"  Spectrogram shape per sample: {X.shape[1:]} "
          f"(128 mel bands x {X.shape[2]} time frames x 1 channel)")
    print(f"  Total samples: {len(y)} "
          f"({np.sum(y == 0)} healthy, {np.sum(y == 1)} pathological)")

    # --- Step 2: Train / test split ---
    print("\nStep 2/4: Splitting into 80% train / 20% test...")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(y_train)} | Test: {len(y_test)}")

    # --- Step 3: Build the CNN ---
    print("\nStep 3/4: Building CNN architecture...")

    # TensorFlow / Keras — imported here so the rest of the project
    # doesn't require TensorFlow unless you run this specific script
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten,
                                         Dense, Dropout, BatchNormalization)
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        # ── Conv block 1: learn low-level spectral patterns ──
        # e.g., "there's a lot of noise in the high-frequency bands"
        Conv2D(32, (3, 3), activation="relu",
               input_shape=X_train.shape[1:]),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # ── Conv block 2: learn mid-level combinations ──
        # e.g., "noise AND pitch instability together"
        Conv2D(64, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # ── Conv block 3: learn high-level voice quality patterns ──
        Conv2D(64, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # ── Classification head ──
        Flatten(),                  # Turn 2D feature maps into a 1D vector
        Dense(128, activation="relu"),
        Dropout(0.5),               # Randomly silence 50% of neurons during
                                    # training to prevent memorisation
        Dense(1, activation="sigmoid"),  # Output: probability of pathological
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0005),  # lower LR to reduce overfitting
        loss="binary_crossentropy",   # Standard loss for binary classification
        metrics=["accuracy"],
    )

    print(f"  Total parameters: {model.count_params():,}")

    # --- Step 4: Train ---
    print("\nStep 4/4: Training (up to 30 epochs, early stopping on val loss)...\n")

    from tensorflow.keras.callbacks import EarlyStopping

    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[EarlyStopping(monitor="val_loss", patience=5,
                                 restore_best_weights=True)],
        verbose=1,
    )

    # --- Evaluation ---
    from sklearn.metrics import classification_report, roc_auc_score, roc_curve

    y_pred_prob = model.predict(X_test).flatten()

    # Find the optimal classification threshold from the ROC curve.
    # The default 0.5 assumes the sigmoid output is well-calibrated, but
    # with small datasets and BatchNorm the outputs can be compressed.
    # Youden's J statistic picks the threshold that maximises
    # (true positive rate − false positive rate).
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = float(thresholds[optimal_idx])
    print(f"\n  Optimal threshold (Youden's J): {optimal_threshold:.4f}")

    y_pred = (y_pred_prob >= optimal_threshold).astype(int)

    print("\n" + "=" * 55)
    print("RESULTS")
    print("=" * 55)
    print(classification_report(
        y_test, y_pred, target_names=["Healthy", "Pathological"]
    ))
    auc = roc_auc_score(y_test, y_pred_prob)
    accuracy = float(np.mean(y_pred == y_test))
    print(f"AUC-ROC: {auc:.4f}")

    # --- Save model and metrics ---
    os.makedirs("model", exist_ok=True)
    model.save("model/spectrogram_cnn.keras")

    cnn_metrics = {
        "accuracy": round(accuracy, 4),
        "auc_roc": round(float(auc), 4),
        "optimal_threshold": round(optimal_threshold, 4),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "spectrogram_shape": list(X_train.shape[1:]),
        "epochs": len(history.history["loss"]),
        "architecture": "3xConv2D + BatchNorm + MaxPool -> Dense(128) -> Sigmoid",
        "note": "Trained on synthetic audio mel-spectrograms, not real clinical data",
    }
    with open("model/cnn_metrics.json", "w") as f:
        json.dump(cnn_metrics, f, indent=2)

    print(f"\nModel saved  -> model/spectrogram_cnn.keras")
    print(f"Metrics saved -> model/cnn_metrics.json")
    print(f"\nDone. The main app uses the tabular pipeline (train_model.py)")
    print(f"for interpretability.  This CNN demonstrates spectrogram-based")
    print(f"deep learning as an alternative approach.")

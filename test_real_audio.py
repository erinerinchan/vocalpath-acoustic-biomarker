"""
Test the VocalPath pipeline on a real audio recording.

Usage:
    python test_real_audio.py                        # uses bundled sample files
    python test_real_audio.py path/to/your/voice.wav # uses your own recording

What this does:
    1. Loads the audio file (WAV, MP3, OGG, FLAC, or M4A)
    2. Extracts the same 26 acoustic features used by the Streamlit app
    3. Runs them through the trained model
    4. Prints a plain-English summary of the result

This script exists to demonstrate that the pipeline works end-to-end on
real audio files — not just on the synthetic tabular data it was trained on.
"""

import sys
import os
import numpy as np
import joblib

from feature_extraction import extract_features_from_file


def main():
    # Determine audio files to test
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        files = ["samples/healthy_vowel.wav", "samples/pathological_vowel.wav"]
        print("No audio file specified — using bundled sample files.\n"
              "Usage: python test_real_audio.py <path_to_your_voice.wav>\n")

    # Load model
    model = joblib.load("model/rf_classifier.joblib")
    feature_names = joblib.load("model/feature_names.joblib")

    for audio_path in files:
        if not os.path.exists(audio_path):
            print(f"File not found: {audio_path}")
            continue

        print(f"\n{'=' * 60}")
        print(f"File: {audio_path}")
        print(f"{'=' * 60}")

        features, duration = extract_features_from_file(audio_path)
        print(f"Duration: {duration:.1f}s")

        # Predict
        feature_vector = np.array([[features.get(f, 0) for f in feature_names]])
        prediction = model.predict(feature_vector)[0]
        probability = model.predict_proba(feature_vector)[0]

        label = "PATHOLOGICAL indicators detected" if prediction == 1 else "HEALTHY (within normal range)"
        confidence = probability[prediction] * 100

        print(f"\nResult: {label}")
        print(f"Confidence: {confidence:.1f}%")
        print(f"P(Healthy) = {probability[0]:.3f}  |  P(Pathological) = {probability[1]:.3f}")

        # Key features
        print(f"\nKey biomarkers:")
        print(f"  Jitter (local):     {features['jitter_local']:.6f}  (healthy ref: < 0.010)")
        print(f"  Shimmer (local):    {features['shimmer_local']:.6f}  (healthy ref: < 0.038)")
        print(f"  HNR:                {features['hnr']:.2f} dB       (healthy ref: > 20 dB)")
        print(f"  F0 mean:            {features['f0_mean']:.1f} Hz")
        print(f"  Spectral flatness:  {features['spectral_flatness']:.6f}")

    print(f"\n{'=' * 60}")
    print("Note: This model was trained on synthetic data. Results on real")
    print("recordings are for demonstration only, not clinical diagnosis.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

"""Unit tests for the VocalPath pipeline.

Run with:  python -m pytest tests/ -v
"""
import os
import sys
import json
import pytest
import numpy as np
import pandas as pd
import joblib

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Data generation ──────────────────────────────────────────────

class TestDataGeneration:
    """Verify synthetic dataset structure and integrity."""

    @pytest.fixture(autouse=True)
    def load_data(self):
        self.df = pd.read_csv(
            os.path.join(os.path.dirname(__file__), "..", "data", "features.csv")
        )

    def test_dataset_has_expected_rows(self):
        assert len(self.df) == 600, f"Expected 600 samples, got {len(self.df)}"

    def test_dataset_has_26_features(self):
        feature_cols = [c for c in self.df.columns if c != "label"]
        assert len(feature_cols) == 26, f"Expected 26 features, got {len(feature_cols)}"

    def test_labels_are_binary(self):
        assert set(self.df["label"].unique()) == {0, 1}

    def test_classes_are_roughly_balanced(self):
        counts = self.df["label"].value_counts()
        ratio = counts.min() / counts.max()
        assert ratio > 0.8, f"Class imbalance ratio {ratio:.2f} is too skewed"

    def test_no_missing_values(self):
        assert self.df.isnull().sum().sum() == 0, "Dataset contains NaN values"

    def test_jitter_is_non_negative(self):
        assert (self.df["jitter_local"] >= 0).all()

    def test_hnr_has_realistic_range(self):
        # HNR should typically be between 0 and 40 dB
        assert self.df["hnr"].min() >= 0
        assert self.df["hnr"].max() <= 45


# ── Model artifacts ──────────────────────────────────────────────

class TestModelArtifacts:
    """Verify trained model files exist and load correctly."""

    MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")

    def test_model_file_exists(self):
        assert os.path.exists(os.path.join(self.MODEL_DIR, "rf_classifier.joblib"))

    def test_feature_names_file_exists(self):
        assert os.path.exists(os.path.join(self.MODEL_DIR, "feature_names.joblib"))

    def test_metrics_file_exists(self):
        assert os.path.exists(os.path.join(self.MODEL_DIR, "metrics.json"))

    def test_model_loads_and_predicts(self):
        model = joblib.load(os.path.join(self.MODEL_DIR, "rf_classifier.joblib"))
        feature_names = joblib.load(os.path.join(self.MODEL_DIR, "feature_names.joblib"))

        # Create a dummy input with 26 features
        dummy = np.zeros((1, len(feature_names)))
        prediction = model.predict(dummy)
        assert prediction.shape == (1,)
        assert prediction[0] in (0, 1)

    def test_model_returns_probabilities(self):
        model = joblib.load(os.path.join(self.MODEL_DIR, "rf_classifier.joblib"))
        feature_names = joblib.load(os.path.join(self.MODEL_DIR, "feature_names.joblib"))

        dummy = np.zeros((1, len(feature_names)))
        proba = model.predict_proba(dummy)
        assert proba.shape == (1, 2)
        assert abs(proba[0].sum() - 1.0) < 1e-6, "Probabilities don't sum to 1"

    def test_metrics_json_has_required_keys(self):
        with open(os.path.join(self.MODEL_DIR, "metrics.json")) as f:
            metrics = json.load(f)
        for key in ["best_model", "accuracy", "precision", "recall", "f1_score", "auc_roc"]:
            assert key in metrics, f"Missing key: {key}"

    def test_feature_names_match_dataset(self):
        feature_names = joblib.load(os.path.join(self.MODEL_DIR, "feature_names.joblib"))
        df = pd.read_csv(
            os.path.join(os.path.dirname(__file__), "..", "data", "features.csv")
        )
        dataset_features = [c for c in df.columns if c != "label"]
        assert list(feature_names) == dataset_features


# ── Feature extraction sanity ────────────────────────────────────

class TestFeatureExtraction:
    """Verify feature extraction produces valid output on sample audio."""

    SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "samples")

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "samples", "healthy_vowel.wav")),
        reason="Sample audio not generated yet"
    )
    def test_extract_features_from_sample(self):
        import librosa
        import parselmouth
        from parselmouth.praat import call

        audio_path = os.path.join(self.SAMPLES_DIR, "healthy_vowel.wav")
        y, sr = librosa.load(audio_path, sr=16000)
        sound = parselmouth.Sound(audio_path)

        # Extract features (inline, mirrors app.py logic)
        features = {}
        pitch = call(sound, "To Pitch", 0.0, 75, 500)
        features["f0_mean"] = call(pitch, "Get mean", 0, 0, "Hertz")
        features["f0_std"] = call(pitch, "Get standard deviation", 0, 0, "Hertz")

        pp = call(sound, "To PointProcess (periodic, cc)", 75, 500)
        features["jitter_local"] = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        features["jitter_rap"] = call(pp, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        features["shimmer_local"] = call([sound, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        features["shimmer_apq3"] = call([sound, pp], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        features["hnr"] = call(harmonicity, "Get mean", 0, 0)

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f"mfcc_{i+1}"] = np.mean(mfccs[i])

        features["spectral_centroid"] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        features["spectral_bandwidth"] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        features["spectral_flatness"] = np.mean(librosa.feature.spectral_flatness(y=y))
        features["spectral_rolloff"] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        features["zcr"] = np.mean(librosa.feature.zero_crossing_rate(y))
        features["rms"] = np.mean(librosa.feature.rms(y=y))

        assert len(features) == 26, f"Expected 26 features, got {len(features)}"
        assert all(isinstance(v, (int, float, np.floating)) for v in features.values())
        assert features["f0_mean"] > 0, "F0 mean should be positive for a vowel"

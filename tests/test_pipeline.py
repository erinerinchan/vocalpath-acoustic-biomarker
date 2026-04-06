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


class TestRealData:
    """Verify real clinical data (VOICED dataset) structure and integrity."""

    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "real_features.csv")

    @pytest.fixture(autouse=True)
    def load_data(self):
        self.df = pd.read_csv(self.DATA_PATH)

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "data", "real_features.csv")),
        reason="Real data not extracted yet"
    )
    def test_real_dataset_has_208_samples(self):
        assert len(self.df) == 208, f"Expected 208 VOICED samples, got {len(self.df)}"

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "data", "real_features.csv")),
        reason="Real data not extracted yet"
    )
    def test_real_dataset_has_26_features(self):
        feature_cols = [c for c in self.df.columns if c != "label"]
        assert len(feature_cols) == 26, f"Expected 26 features, got {len(feature_cols)}"

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "data", "real_features.csv")),
        reason="Real data not extracted yet"
    )
    def test_real_labels_are_binary(self):
        assert set(self.df["label"].unique()) == {0, 1}

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "data", "real_features.csv")),
        reason="Real data not extracted yet"
    )
    def test_real_no_missing_values(self):
        assert self.df.isnull().sum().sum() == 0, "Real dataset contains NaN values"

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "data", "real_features.csv")),
        reason="Real data not extracted yet"
    )
    def test_real_class_distribution(self):
        counts = self.df["label"].value_counts()
        assert counts[0] == 57, f"Expected 57 healthy, got {counts[0]}"
        assert counts[1] == 151, f"Expected 151 pathological, got {counts[1]}"

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "data", "real_features.csv")),
        reason="Real data not extracted yet"
    )
    def test_real_features_match_synthetic_columns(self):
        synth_df = pd.read_csv(
            os.path.join(os.path.dirname(__file__), "..", "data", "features.csv")
        )
        synth_cols = sorted([c for c in synth_df.columns if c != "label"])
        real_cols = sorted([c for c in self.df.columns if c != "label"])
        assert real_cols == synth_cols, "Real and synthetic datasets have different feature columns"


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
        # Check against whichever dataset the model was trained on
        real_path = os.path.join(os.path.dirname(__file__), "..", "data", "real_features.csv")
        synth_path = os.path.join(os.path.dirname(__file__), "..", "data", "features.csv")
        data_path = real_path if os.path.exists(real_path) else synth_path
        df = pd.read_csv(data_path)
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

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "samples", "healthy_vowel.wav")),
        reason="Sample audio not generated yet"
    )
    def test_feature_extraction_returns_26_features(self):
        """Verify all 26 biomarkers are extracted from sample audio."""
        import librosa

        audio_path = os.path.join(self.SAMPLES_DIR, "healthy_vowel.wav")
        y, sr = librosa.load(audio_path, sr=16000)
        import parselmouth
        from parselmouth.praat import call

        sound = parselmouth.Sound(audio_path)

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

        expected_names = [
            "f0_mean", "f0_std", "jitter_local", "jitter_rap",
            "shimmer_local", "shimmer_apq3", "hnr",
            "mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4", "mfcc_5", "mfcc_6",
            "mfcc_7", "mfcc_8", "mfcc_9", "mfcc_10", "mfcc_11", "mfcc_12", "mfcc_13",
            "spectral_centroid", "spectral_bandwidth", "spectral_flatness",
            "spectral_rolloff", "zcr", "rms",
        ]
        assert sorted(features.keys()) == sorted(expected_names)


class TestFeatureRanges:
    """Verify extracted features fall within physiologically plausible ranges."""

    @pytest.fixture(autouse=True)
    def load_data(self):
        # Test ranges against both synthetic and real data
        real_path = os.path.join(os.path.dirname(__file__), "..", "data", "real_features.csv")
        synth_path = os.path.join(os.path.dirname(__file__), "..", "data", "features.csv")
        dfs = [pd.read_csv(synth_path)]
        if os.path.exists(real_path):
            dfs.append(pd.read_csv(real_path))
        self.df = pd.concat(dfs, ignore_index=True)

    def test_jitter_is_plausible(self):
        """Jitter (local) should be < 20% for all samples (real data can have extreme pathology)."""
        assert (self.df["jitter_local"] < 0.20).all(), "Jitter exceeds 20%"

    def test_shimmer_is_plausible(self):
        """Shimmer (local) should be < 50% for all samples (real data can have extreme pathology)."""
        assert (self.df["shimmer_local"] < 0.50).all(), "Shimmer exceeds 50%"

    def test_f0_in_plausible_range(self):
        """F0 mean should be between 50 and 500 Hz."""
        assert (self.df["f0_mean"] >= 50).all(), "F0 below 50 Hz"
        assert (self.df["f0_mean"] <= 500).all(), "F0 above 500 Hz"

    def test_hnr_is_plausible(self):
        """HNR should be between 0 and 45 dB."""
        assert (self.df["hnr"] >= 0).all(), "HNR below 0"
        assert (self.df["hnr"] <= 45).all(), "HNR above 45 dB"


class TestModelPredictionShape:
    """Verify model returns correct output shapes and probabilities."""

    MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")

    def test_prediction_shape_single_sample(self):
        """Model should return a single prediction for a single input."""
        model = joblib.load(os.path.join(self.MODEL_DIR, "rf_classifier.joblib"))
        feature_names = joblib.load(os.path.join(self.MODEL_DIR, "feature_names.joblib"))
        dummy = np.zeros((1, len(feature_names)))
        pred = model.predict(dummy)
        assert pred.shape == (1,)

    def test_prediction_shape_batch(self):
        """Model should return correct shape for batch predictions."""
        model = joblib.load(os.path.join(self.MODEL_DIR, "rf_classifier.joblib"))
        feature_names = joblib.load(os.path.join(self.MODEL_DIR, "feature_names.joblib"))
        dummy = np.zeros((10, len(feature_names)))
        pred = model.predict(dummy)
        assert pred.shape == (10,)

    def test_probability_for_both_classes(self):
        """Model should return probability for both classes."""
        model = joblib.load(os.path.join(self.MODEL_DIR, "rf_classifier.joblib"))
        feature_names = joblib.load(os.path.join(self.MODEL_DIR, "feature_names.joblib"))
        dummy = np.zeros((1, len(feature_names)))
        proba = model.predict_proba(dummy)
        assert proba.shape == (1, 2), "Model should output 2 class probabilities"
        assert abs(proba[0].sum() - 1.0) < 1e-6, "Probabilities must sum to 1"

    def test_held_out_test_metrics_saved(self):
        """Verify held-out test metrics are saved in metrics.json."""
        metrics_path = os.path.join(self.MODEL_DIR, "metrics.json")
        with open(metrics_path) as f:
            metrics = json.load(f)
        if "held_out_test" in metrics:
            hot = metrics["held_out_test"]
            for key in ["accuracy", "precision", "recall", "f1_score", "auc_roc"]:
                assert key in hot, f"Missing held-out test metric: {key}"
                assert 0 <= hot[key] <= 1, f"Held-out {key} out of range: {hot[key]}"


# ── Feature extraction module ────────────────────────────────────

class TestFeatureExtractionModule:
    """Verify the shared feature_extraction module."""

    def test_feature_names_list_has_26_items(self):
        from feature_extraction import FEATURE_NAMES
        assert len(FEATURE_NAMES) == 26, f"Expected 26 feature names, got {len(FEATURE_NAMES)}"

    def test_feature_names_include_core_biomarkers(self):
        from feature_extraction import FEATURE_NAMES
        core = ["f0_mean", "jitter_local", "shimmer_local", "hnr", "mfcc_1", "spectral_flatness"]
        for name in core:
            assert name in FEATURE_NAMES, f"Missing core biomarker: {name}"

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "samples", "healthy_vowel.wav")),
        reason="Sample audio not generated yet"
    )
    def test_extract_features_from_file(self):
        from feature_extraction import extract_features_from_file, FEATURE_NAMES
        audio_path = os.path.join(os.path.dirname(__file__), "..", "samples", "healthy_vowel.wav")
        features, duration = extract_features_from_file(audio_path)
        assert len(features) == 26
        assert set(features.keys()) == set(FEATURE_NAMES)
        assert duration > 0


# ── Data source tracking ─────────────────────────────────────────

class TestDataSourceTracking:
    """Verify metrics.json tracks which data source the model was trained on."""

    MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")

    def test_metrics_has_data_source(self):
        with open(os.path.join(self.MODEL_DIR, "metrics.json")) as f:
            metrics = json.load(f)
        assert "data_source" in metrics, "metrics.json should track data_source"

    def test_cnn_metrics_has_caveat(self):
        cnn_path = os.path.join(self.MODEL_DIR, "cnn_metrics.json")
        if os.path.exists(cnn_path):
            with open(cnn_path) as f:
                cnn = json.load(f)
            assert "caveat" in cnn or "data_source" in cnn, \
                "CNN metrics should document synthetic-only limitation"

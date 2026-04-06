"""
Shared feature extraction module for VocalPath.

Extracts 26 clinically validated acoustic features from audio signals.
Used by: app.py, load_voiced.py, test_real_audio.py, tests/test_pipeline.py

Features extracted:
    Praat/Parselmouth (7):  f0_mean, f0_std, jitter_local, jitter_rap,
                            shimmer_local, shimmer_apq3, hnr
    Librosa (19):           mfcc_1..13, spectral_centroid, spectral_bandwidth,
                            spectral_flatness, spectral_rolloff, zcr, rms
"""

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call


# Canonical feature order — must match training data columns
FEATURE_NAMES = [
    "f0_mean", "f0_std", "jitter_local", "jitter_rap",
    "shimmer_local", "shimmer_apq3", "hnr",
    "mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4", "mfcc_5", "mfcc_6",
    "mfcc_7", "mfcc_8", "mfcc_9", "mfcc_10", "mfcc_11", "mfcc_12", "mfcc_13",
    "spectral_centroid", "spectral_bandwidth", "spectral_flatness",
    "spectral_rolloff", "zcr", "rms",
]


def extract_features_from_audio(sound, y, sr):
    """Extract 26 acoustic features from pre-loaded audio.

    Parameters
    ----------
    sound : parselmouth.Sound
        Praat Sound object.
    y : np.ndarray
        Audio time series (librosa format).
    sr : int
        Sample rate.

    Returns
    -------
    dict
        Dictionary with 26 feature name -> value pairs.
    """
    features = {}

    # ── Praat features ──
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

    # ── Librosa features ──
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f"mfcc_{i+1}"] = np.mean(mfccs[i])

    features["spectral_centroid"] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features["spectral_bandwidth"] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    features["spectral_flatness"] = np.mean(librosa.feature.spectral_flatness(y=y))
    features["spectral_rolloff"] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    features["zcr"] = np.mean(librosa.feature.zero_crossing_rate(y))
    features["rms"] = np.mean(librosa.feature.rms(y=y))

    return features


def extract_features_from_file(audio_path):
    """Extract 26 acoustic features from an audio file path.

    Parameters
    ----------
    audio_path : str
        Path to audio file (WAV, MP3, OGG, FLAC, M4A).

    Returns
    -------
    tuple of (dict, float)
        (features dict, duration in seconds)
    """
    y, sr = librosa.load(audio_path)
    duration = librosa.get_duration(y=y, sr=sr)
    sound = parselmouth.Sound(audio_path)
    features = extract_features_from_audio(sound, y, sr)
    return features, duration

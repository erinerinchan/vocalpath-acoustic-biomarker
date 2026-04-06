"""Generate a synthetic sustained vowel .wav for testing the app."""

import numpy as np
import soundfile as sf
import os


def generate_vowel(duration=3.0, sr=22050, f0=180, noise_level=0.02):
    """Synthesize a sustained 'ah' vowel with harmonics."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Fundamental + harmonics (vocal fold vibration)
    signal = np.zeros_like(t)
    harmonics = [1.0, 0.5, 0.3, 0.2, 0.1, 0.07, 0.04]
    for i, amp in enumerate(harmonics, 1):
        signal += amp * np.sin(2 * np.pi * f0 * i * t)

    # Formant-like filtering (simple resonances for /a/ vowel)
    # F1 ~730 Hz, F2 ~1090 Hz, F3 ~2440 Hz
    for formant_freq, formant_amp in [(730, 0.3), (1090, 0.15), (2440, 0.08)]:
        signal += formant_amp * np.sin(2 * np.pi * formant_freq * t)

    # Add slight breathiness (noise)
    signal += noise_level * np.random.randn(len(t))

    # Amplitude envelope (gentle onset/offset)
    envelope = np.ones_like(t)
    attack = int(0.1 * sr)
    release = int(0.15 * sr)
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-release:] = np.linspace(1, 0, release)
    signal *= envelope

    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8
    return signal.astype(np.float32)


if __name__ == "__main__":
    os.makedirs("samples", exist_ok=True)

    # Healthy-sounding voice
    healthy = generate_vowel(duration=3.0, f0=180, noise_level=0.02)
    sf.write("samples/healthy_vowel.wav", healthy, 22050)

    # Pathological-sounding voice (higher noise, irregular pitch)
    pathological = generate_vowel(duration=3.0, f0=165, noise_level=0.08)
    # Add shimmer-like amplitude modulation
    shimmer = 1.0 + 0.06 * np.random.randn(len(pathological))
    pathological = (pathological * shimmer).astype(np.float32)
    pathological = pathological / np.max(np.abs(pathological)) * 0.8
    sf.write("samples/pathological_vowel.wav", pathological, 22050)

    print("Generated: samples/healthy_vowel.wav")
    print("Generated: samples/pathological_vowel.wav")

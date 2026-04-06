# Data Sources

## Synthetic (`generate_demo_data.py`)

- **600 samples** (300 healthy, 300 pathological) generated from published clinical ranges
- Distributions based on:
  - Teixeira et al. (2013) — jitter, shimmer, HNR parameters
  - Godino-Llorente et al. (2006) — dimensionality reduction of voice quality features
  - Martinez et al. (2012) — voice pathology detection on SVD
- Realistic inter-class overlap targeting ~85% classification accuracy
- Inter-feature correlations via a latent severity model
- 5% label noise to simulate diagnostic ambiguity
- Used for initial development, pipeline testing, and EDA

## Saarbrücken Voice Database (SVD)

- Real clinical recordings from the University of Saarland
- Sustained vowels /a/, /i/, /u/ at normal, high, and low pitch
- Gold-standard labels from ENT diagnosis (laryngoscopy-confirmed)
- Available at: https://stimmdb.coli.uni-saarland.de/
- **Status:** Not yet integrated — listed as a future validation step

## Feature Schema

Each row in `features.csv` contains 26 acoustic features + 1 label column:

| Feature | Category | Description |
|---------|----------|-------------|
| `f0_mean` | Perturbation | Mean fundamental frequency (Hz) |
| `f0_std` | Perturbation | F0 standard deviation (Hz) |
| `jitter_local` | Perturbation | Cycle-to-cycle pitch variation |
| `jitter_rap` | Perturbation | Relative average perturbation |
| `shimmer_local` | Perturbation | Cycle-to-cycle amplitude variation |
| `shimmer_apq3` | Perturbation | 3-point amplitude perturbation quotient |
| `hnr` | Perturbation | Harmonics-to-noise ratio (dB) |
| `mfcc_1` — `mfcc_13` | Spectral | Mel-frequency cepstral coefficients |
| `spectral_centroid` | Spectral | Perceived brightness |
| `spectral_bandwidth` | Spectral | Spectral spread |
| `spectral_flatness` | Spectral | Noise-like vs tone-like |
| `spectral_rolloff` | Spectral | High-frequency energy boundary |
| `zcr` | Temporal | Zero-crossing rate |
| `rms` | Temporal | Root mean square energy |
| `label` | Target | 0 = Healthy, 1 = Pathological |

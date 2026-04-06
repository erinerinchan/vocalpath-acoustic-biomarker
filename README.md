# VocalPath: Acoustic Biomarker Analysis for Voice Pathology

A machine learning tool for voice pathology screening through acoustic biomarker extraction and classification. Built with Python, Streamlit, Praat/Parselmouth, and Librosa.

![VocalPath Demo](screenshots/demo.png)

## Overview

VocalPath extracts **26 clinically validated acoustic features** from voice recordings and classifies them as healthy or potentially pathological using a trained classifier. The tool provides interactive visualizations comparing extracted features against healthy baselines.

The model is trained on **real clinical recordings** from the VOICED dataset — 208 voice samples collected at the University of Naples Federico II hospital, with diagnoses confirmed by laryngoscopy.

Voice changes — particularly persistent hoarseness — are among the earliest symptoms of **laryngeal cancer**, one of the most common head-and-neck cancers (American Cancer Society, 2024). An automated screening tool could flag at-risk individuals for timely ENT referral and laryngoscopy, potentially improving early detection rates in under-served communities.

**This is a research prototype and is not intended for clinical diagnosis.**

> **[▶ Try the live demo on Streamlit Cloud](https://vocalpath-acoustic-biomarker.streamlit.app/)**

## Features

- **26 acoustic biomarkers** including jitter, shimmer, HNR, MFCCs, spectral features
- **Multi-model comparison** (Random Forest, SVM, Logistic Regression) with 5-fold stratified CV
- **SMOTE oversampling comparison** — evaluates whether synthetic minority oversampling improves classification
- **Held-out test set evaluation** (80/20 stratified split) for honest performance estimates
- **SHAP explanations** showing which features drove each individual prediction
- **Precision-recall analysis** with clinical discussion of why recall matters in cancer screening
- **Bootstrap confidence intervals** (1,000 iterations) for all metrics
- **Learning curves** to assess whether more data would improve performance
- **Interactive visualizations**: waveform, MFCC spectrogram, radar chart, confusion matrix, ROC curve, PR curve
- **Downloadable HTML report** for each analysis
- **Real audio testing** script to run voice recordings through the pipeline
- **Input validation** with minimum duration (2s) and silence detection
- **Methodology documentation** with clinical references

## Quick Start

```bash
# Clone and setup
git clone https://github.com/erinerinchan/vocalpath-acoustic-biomarker.git
cd vocalpath-acoustic-biomarker
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Generate data and train model
python generate_demo_data.py
python train_model.py

# Generate test audio files (optional)
python generate_sample_audio.py

# Launch the app
streamlit run app.py
```

## Project Structure

```
vocal-path-ai/
├── app.py                    # Streamlit web application
├── feature_extraction.py     # Shared 26-feature extraction module
├── load_voiced.py            # VOICED clinical dataset ingestion
├── generate_demo_data.py     # Synthetic data generation with realistic overlap
├── train_model.py            # Multi-model training, SMOTE comparison, evaluation
├── evaluate_model.py         # Standalone model evaluation with threshold analysis
├── visualize_results.py      # Publication-quality plots (radar, importance, comparison)
├── generate_sample_audio.py  # Synthetic test .wav generator
├── generate_readme_demo.py   # Generates README demo image
├── train_spectrogram_cnn.py  # Mel-spectrogram CNN (deep learning, optional)
├── test_real_audio.py        # Test pipeline on real audio recordings
├── CHANGELOG.md              # Version history and changes
├── tests/
│   └── test_pipeline.py      # Unit tests for core pipeline
├── .streamlit/
│   └── config.toml            # Streamlit Cloud theme config
├── requirements.txt
├── .gitignore
├── data/
│   ├── features.csv          # Synthetic feature dataset (600 samples)
│   ├── real_features.csv     # Real clinical features (VOICED, 208 samples)
│   ├── real_features_full.csv # Real features with demographics
│   ├── voiced/               # VOICED clinical recordings (not in git)
│   └── README.md             # Data provenance and schema documentation
├── model/
│   ├── rf_classifier.joblib  # Trained model
│   ├── feature_names.joblib  # Feature column names
│   ├── metrics.json          # Cross-validation results + SMOTE comparison
│   ├── evaluation_report.json # Standalone evaluation report
│   ├── confusion_matrix.csv  # CV confusion matrix
│   ├── roc_data.csv          # ROC curve data
│   ├── pr_curve.csv          # Precision-recall curve data
│   ├── learning_curve.csv    # Learning curve data (F1 vs sample size)
│   └── feature_importance.csv
├── notebooks/
│   └── eda.ipynb             # Exploratory data analysis
├── screenshots/
│   └── demo.png              # README demo image
├── samples/
│   ├── healthy_vowel.wav     # Test audio (synthetic)
│   └── pathological_vowel.wav
└── README.md
```

## Methodology

### Acoustic Features

| Category | Features | Tool |
|---|---|---|
| Perturbation | Jitter (local, RAP), Shimmer (local, APQ3), HNR | Praat/Parselmouth |
| Spectral | MFCCs (1-13), Centroid, Bandwidth, Flatness, Rolloff | Librosa |
| Temporal | ZCR, RMS, F0 (mean, std) | Both |

### Classification

Three models compared via 5-fold stratified cross-validation:
- Random Forest (100 trees, max depth 10)
- SVM (RBF kernel)
- Logistic Regression

Best model selected by F1 score. All models use StandardScaler preprocessing and balanced class weights.

### Evaluation Methodology

- **Stratified 80/20 train/test split** held out before any cross-validation to prevent data leakage
- **5-fold stratified CV** on the training set for model selection
- **Held-out test set** for final, unbiased performance estimate
- **Precision-recall curves** with clinical discussion: in cancer screening, recall (sensitivity) is prioritised over precision — missing a sick person is worse than a false alarm
- **Bootstrap confidence intervals** (1,000 resamples) provide honest uncertainty estimates rather than single-point accuracy claims
- **Learning curves** show model performance vs. training set size, indicating whether more data would help
- **SHAP explanations** for per-prediction interpretability — shows which features drove each individual result
- **Cohen's d effect sizes** and Mann-Whitney U tests quantify feature discriminative power with statistical rigour

## Results

### Real Clinical Data (VOICED Dataset, N=208)

The model is now trained and evaluated on the **VOICED clinical dataset** — 208 real voice recordings (57 healthy, 151 pathological) from the University of Naples Federico II hospital, with diagnoses confirmed by laryngoscopy under the SIFEL protocol.

#### Model Comparison (5-Fold CV on Training Set, N=166)

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|-------|----------|-----------|--------|-----|---------|
| **Random Forest** | 0.729 | 0.751 | **0.942** | **0.835** | 0.688 |
| SVM (RBF) | 0.711 | **0.855** | 0.735 | 0.786 | **0.724** |
| Logistic Regression | 0.687 | 0.831 | 0.719 | 0.769 | 0.718 |

*Selected model: Random Forest (highest F1). High recall (94%) is desirable for a screening tool.*

#### SMOTE Oversampling Comparison

| Configuration | F1 | Recall | Accuracy |
|--------------|-----|--------|----------|
| **Without SMOTE** (balanced class weights) | **0.835** | **0.942** | **0.729** |
| With SMOTE | 0.819 | 0.826 | — |

*SMOTE does not improve performance — balanced class weights already handle the 57:151 imbalance effectively. The balanced-weights approach achieves higher recall, which is critical for screening.*

#### Held-Out Test Set (20% Stratified Split, N=42)

| Metric | Value |
|--------|-------|
| Accuracy | 0.690 |
| Precision | 0.730 |
| Recall | 0.900 |
| F1 | 0.806 |
| AUC-ROC | 0.637 |

#### 95% Bootstrap Confidence Intervals (1,000 iterations)

| Metric | Mean | 95% CI |
|--------|------|--------|
| Accuracy | 0.734 | [0.663, 0.801] |
| F1 | 0.838 | [0.786, 0.882] |
| Recall | 0.950 | [0.908, 0.984] |
| Precision | 0.751 | [0.678, 0.813] |
| AUC-ROC | 0.676 | [0.576, 0.766] |

> **Why accuracy is lower than synthetic data (~69% vs ~89%):** Real clinical recordings contain noise, microphone variability, accent differences, and subtle pathology that synthetic data cannot fully model. The high recall (90% on test set) means the model successfully flags most pathological voices — the primary goal in a screening context. Lower precision means more false alarms, but in cancer screening, a false alarm (unnecessary laryngoscopy) is far less harmful than a missed case (delayed diagnosis).

### Data

Current model is trained on the **VOICED clinical dataset** (Cesari et al., 2018):
- 208 real voice recordings from University of Naples Federico II hospital
- 57 healthy speakers, 151 with clinically verified voice pathologies
- Diagnoses: hyperkinetic/hypokinetic dysphonia, reflux laryngitis, vocal fold nodules/polyps/paralysis
- Sustained vowel 'a', 5 seconds, 8 kHz, recorded via mobile device in clinical setting
- All diagnoses confirmed by medical experts using the SIFEL protocol

Synthetic data (600 samples) is also available in `data/features.csv` for comparison.

## Clinical Validity & Limitations

### Technical Validity
- Model trained and evaluated on **real clinical recordings** from the VOICED dataset (208 patients, University of Naples)
- Best model achieves 73% accuracy, 84% F1, and 94% recall via 5-fold stratified cross-validation
- **95% bootstrap confidence intervals**: Accuracy [0.663, 0.801], F1 [0.786, 0.882], AUC-ROC [0.576, 0.766]
- High recall (90% on held-out test) confirms suitability as a screening tool
- Feature extraction uses Praat and Librosa — standard tools in published voice-pathology research

### Clinical Validity Gaps
- **Single-site data** — model trained on recordings from one hospital; generalizability unverified
- **No external dataset testing** — not evaluated on SVD, MEEI, or other independent clinical corpora
- **Class imbalance** — 57 healthy vs 151 pathological (mitigated by balanced class weights)
- **No demographic sub-group analysis** — performance may vary by age, sex, accent, or language
- **No prospective study** — not tested alongside ENT specialists in real-world settings
- **Recording conditions uncontrolled** — consumer microphones and ambient noise introduce untested variability

### What Full Clinical Validation Would Require
1. IRB/ethics approval for patient voice collection
2. Gold-standard labels via laryngoscopy (following STARD guidelines)
3. External validation on ≥1 independent clinical dataset
4. Sub-group analysis for bias across demographics
5. Transparent reporting per TRIPOD checklist

## Benefits & Risks of AI in Voice Screening

### Potential Benefits
- **Early detection** — voice changes can appear before other symptoms
- **Accessibility** — smartphone-based screening for under-served communities
- **Reduced clinician workload** — pre-screening prioritises specialist referrals
- **Objective measurement** — quantitative features complement subjective evaluation
- **Patient empowerment** — longitudinal self-monitoring with shareable results

### Potential Risks
- **False negatives** — missed pathology could delay treatment
- **False positives** — unnecessary anxiety and specialist visits
- **Over-reliance** — users may trust the tool beyond its validation level
- **Data privacy** — voice is biometric data; storage/transmission must comply with GDPR/HIPAA
- **Bias and fairness** — under-represented groups may receive worse predictions
- **Regulatory gap** — deployment without CE/FDA clearance poses safety and legal risks

### Ethical Safeguards in This Prototype
- Clear "research prototype" disclaimers shown before every result
- No user recordings are stored or transmitted
- Limitation warnings advise professional consultation
- All model performance data openly reported

## Spectrogram-Based Deep Learning (Experimental)

In addition to the tabular pipeline, `train_spectrogram_cnn.py` implements an alternative approach: a CNN that classifies voice pathology directly from **mel-spectrogram images**. This preserves temporal patterns that frame-averaged features discard.

- Architecture: 3×Conv2D + BatchNorm + MaxPool → Dense(128) → Sigmoid
- Input: 128-band mel-spectrograms from 3-second synthetic vowels
- Optional dependency: `pip install tensorflow`

The tabular pipeline is used in the app for interpretability; the CNN demonstrates deeper audio-ML capability.

> **Important caveat on CNN metrics:** The CNN achieves 100% accuracy on **synthetic** audio where pathological markers are programmatically injected. This is a pipeline validation artifact, NOT a clinical performance claim. On real clinical data, the tabular pipeline achieves ~69% accuracy — expect similar or worse from a CNN on real recordings. The CNN has **not** been evaluated on VOICED or any clinical dataset.

## Future Directions

- **Self-supervised models**: Fine-tuning wav2vec 2.0 or HuBERT on clinical voice data for richer representations
- **Whisper embeddings**: Using OpenAI Whisper's intermediate features for voice quality analysis
- **Real clinical validation**: Testing on SVD, MEEI, or hospital-collected laryngoscopy-confirmed samples
- **Longitudinal monitoring**: Tracking voice changes over weeks to detect gradual deterioration (relevant for early cancer)
- **mHealth deployment**: Smartphone app with periodic voice sampling and trend alerts

## References

1. Cesari, U. et al. (2018). "A new database of healthy and pathological voices." *Computers & Electrical Engineering*, 68, 310-321.
2. Teixeira, J.P. et al. (2013). "Vocal Acoustic Analysis — Jitter, Shimmer and HNR Parameters." *Procedia Technology*, 9.
3. Godino-Llorente, J.I. et al. (2006). "Dimensionality Reduction of a Pathological Voice Quality Feature Set." *IEEE Trans. BME*, 53(10).
4. Martinez, D. et al. (2012). "Voice Pathology Detection on the Saarbrucken Voice Database."
5. Boersma, P. & Weenink, D. (2023). Praat: doing phonetics by computer.
6. American Cancer Society (2024). "Signs and Symptoms of Laryngeal and Hypopharyngeal Cancers."
7. Baevski, A. et al. (2020). "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations." *NeurIPS*.
8. Hsu, W.-N. et al. (2021). "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction." *IEEE/ACM Trans. ASLP*.
9. Hemmerling, D. et al. (2016). "Voice data mining for laryngeal pathology assessment." *Computers in Biology and Medicine*, 69.

## License

MIT

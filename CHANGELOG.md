# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2026-04-06

### Added
- **Held-out test set evaluation** — 80/20 stratified split before any CV to prevent data leakage
- **SHAP explanations** — per-prediction feature importance waterfall chart in the Analysis tab
- **Improved input validation** — minimum 2-second duration check and silence/low-energy detection
- **Data provenance documentation** — `data/README.md` with source descriptions and feature schema
- **Expanded test suite** — tests for feature ranges, prediction shapes, held-out metrics
- **Held-out test results display** in the "How Accurate Is This Tool?" tab
- **Results comparison table** in README

### Changed
- `train_model.py` now splits data 80/20 before cross-validation; CV runs on training set only
- Audio upload error messages are more descriptive with troubleshooting tips
- README updated with evaluation methodology improvements

## [0.1.0] - 2026-03-XX

### Added
- Initial synthetic data pipeline (`generate_demo_data.py`)
- Multi-model comparison: Random Forest, SVM, Logistic Regression
- 5-fold stratified cross-validation with bootstrap confidence intervals
- Precision-recall analysis with clinical screening discussion
- Learning curves for sample-size analysis
- Streamlit app with audio upload, recording, waveform, MFCC spectrogram, radar chart
- Downloadable HTML analysis reports
- Spectrogram CNN (experimental, `train_spectrogram_cnn.py`)
- Real audio testing script (`test_real_audio.py`)
- Unit tests for pipeline (`tests/test_pipeline.py`)
- Exploratory data analysis notebook (`notebooks/eda.ipynb`)
- Clinical validity & limitations documentation
- Benefits & risks of AI in voice screening

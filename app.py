import streamlit as st
import librosa
import librosa.display
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import joblib
import json
import os
import io
import tempfile
import soundfile as sf
from datetime import datetime
from audio_recorder_streamlit import audio_recorder

# --- PAGE CONFIG ---
st.set_page_config(page_title="VocalPath | Acoustic Biomarker Analysis", page_icon="🔬", layout="wide")

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    h1, h2, h3 { color: #1e3a8a; font-family: 'Segoe UI', sans-serif; }
    .stButton>button {
        width: 100%; border-radius: 5px; height: 3em;
        background-color: #007bff; color: white; border: none;
    }
    .stButton>button:hover { background-color: #0056b3; }
    .metric-card {
        background: white; padding: 20px; border-radius: 10px;
        border: 1px solid #e0e0e0; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    .disclaimer {
        background: #fff3cd; padding: 15px; border-radius: 8px;
        border-left: 4px solid #ffc107; margin-top: 20px;
    }

    /* ── Responsive ── */
    /* Plotly and images should never overflow */
    .stPlotlyChart, .stImage, .js-plotly-plot {
        max-width: 100% !important;
        overflow-x: auto !important;
    }

    /* Tablets and below: stack Streamlit columns vertically */
    @media (max-width: 768px) {
        [data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
        }
        [data-testid="stHorizontalBlock"] > div {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 0 !important;
        }
        /* Slightly reduce heading sizes */
        h1 { font-size: 1.5rem !important; }
        h2 { font-size: 1.25rem !important; }
        h3 { font-size: 1.1rem !important; }
        /* Cards need breathing room */
        .metric-card { padding: 14px; }
        /* Metrics row: let them wrap nicely */
        [data-testid="stMetricValue"] { font-size: 1.2rem !important; }
    }

    /* Small phones */
    @media (max-width: 480px) {
        h1 { font-size: 1.25rem !important; }
        h2 { font-size: 1.1rem !important; }
        h3 { font-size: 1rem !important; }
        .metric-card { padding: 12px; }
        .metric-card h2 { font-size: 1rem !important; }
    }

    /* Mobile-only sidebar reminder */
    .mobile-sidebar-hint { display: none; }
    @media (max-width: 768px) {
        .mobile-sidebar-hint {
            display: block;
            background: #e3f2fd;
            border: 1px solid #90caf9;
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 16px;
            text-align: center;
        }
        .mobile-sidebar-hint .hint-icon { font-size: 2rem; margin-bottom: 6px; }
    }
        .disclaimer { padding: 10px; font-size: 0.85em; }
        /* Sidebar file uploader */
        section[data-testid="stSidebar"] { min-width: 0 !important; }
    }
</style>
""", unsafe_allow_html=True)


def extract_features(sound, y, sr):
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
    return features


def generate_report_html(features, prediction, probability, duration):
    label = "Pathological Indicators Detected" if prediction == 1 else "Within Normal Range"
    color = "#d32f2f" if prediction == 1 else "#2e7d32"
    confidence = probability[prediction] * 100
    rows = "".join(
        f"<tr><td>{k}</td><td>{v:.6f}</td></tr>" for k, v in features.items()
    )
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>VocalPath Report</title>
<style>
body {{ font-family: 'Segoe UI', sans-serif; max-width: 800px; margin: 40px auto; padding: 0 16px; color: #333; }}
h1 {{ color: #1e3a8a; border-bottom: 2px solid #007bff; padding-bottom: 10px; font-size: clamp(1.3rem, 4vw, 1.8rem); }}
.result {{ background: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid {color}; margin: 20px 0; }}
.result h2 {{ color: {color}; margin: 0 0 10px 0; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background: #007bff; color: white; }}
tr:nth-child(even) {{ background: #f8f9fa; }}
.disclaimer {{ background: #fff3cd; padding: 15px; border-radius: 8px; margin-top: 30px; font-size: 0.85em; }}
@media (max-width: 600px) {{
    body {{ margin: 16px auto; }}
    table {{ font-size: 0.85em; }}
    th, td {{ padding: 6px 4px; }}
}}
</style></head><body>
<h1>VocalPath: Acoustic Biomarker Report</h1>
<p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
<p>Audio duration: {duration:.1f}s</p>
<div class="result">
<h2>{label}</h2>
<p>Confidence: {confidence:.1f}% &nbsp;|&nbsp; P(Healthy)={probability[0]:.3f} &nbsp;|&nbsp; P(Pathological)={probability[1]:.3f}</p>
</div>
<h3>Extracted Features ({len(features)})</h3>
<table><tr><th>Feature</th><th>Value</th></tr>{rows}</table>
<div class="disclaimer"><strong>Disclaimer:</strong> This report is generated by a research prototype trained on synthetic data.
It is not intended for clinical diagnosis. Consult a qualified medical professional for any health concerns.</div>
</body></html>"""


@st.cache_resource
def load_model():
    model = joblib.load("model/rf_classifier.joblib")
    feature_names = joblib.load("model/feature_names.joblib")
    with open("model/metrics.json") as f:
        metrics = json.load(f)
    importance = pd.read_csv("model/feature_importance.csv")
    return model, feature_names, metrics, importance


@st.cache_data
def load_eval_artifacts():
    cm = pd.read_csv("model/confusion_matrix.csv", index_col=0)
    roc = pd.read_csv("model/roc_data.csv")
    pr = pd.read_csv("model/pr_curve.csv") if os.path.exists("model/pr_curve.csv") else None
    lc = pd.read_csv("model/learning_curve.csv") if os.path.exists("model/learning_curve.csv") else None
    return cm, roc, pr, lc


# --- HEADER ---
st.title("🔬 VocalPath: Acoustic Biomarker Analysis")
st.markdown("**Voice pathology screening through machine learning and acoustic feature analysis**")

tab_analysis, tab_methodology, tab_model = st.tabs(["Analysis", "Methodology", "How Accurate Is This Tool?"])

# --- SIDEBAR ---
with st.sidebar:
    st.header("Voice Sample")
    input_method = st.radio("Choose input method:", ["Upload a file", "Record your voice"], horizontal=True)

    audio_data = None
    if input_method == "Upload a file":
        uploaded_file = st.file_uploader("Upload a sustained vowel", type=["wav", "mp3", "ogg", "flac", "m4a"])
        if uploaded_file is not None:
            audio_data = uploaded_file
    else:
        st.markdown("Click the microphone to start recording, click again to stop:")
        recorded_bytes = audio_recorder(
            text="",
            recording_color="#d32f2f",
            neutral_color="#1e3a8a",
            icon_size="2x",
            pause_threshold=3.0,
        )
        if recorded_bytes:
            audio_data = io.BytesIO(recorded_bytes)
            st.audio(recorded_bytes, format="audio/wav")

    st.markdown("---")
    st.markdown("**Recording Guidelines**")
    st.markdown("- Sustained 'Ah' vowel, 3\u20135 seconds\n- Quiet environment\n- Consistent volume\n- Formats: WAV, MP3, OGG, FLAC, M4A")
    st.markdown("""
    <div class='disclaimer'>
        <strong>\u26a0\ufe0f Research Prototype</strong><br>
        This tool is for academic demonstration only and is <strong>not</strong> intended
        for clinical diagnosis. No audio data is stored; all processing occurs in-memory.
    </div>
    """, unsafe_allow_html=True)

# --- ANALYSIS TAB ---
with tab_analysis:
    if audio_data is not None:
        try:
            model, feature_names, metrics, importance = load_model()
        except FileNotFoundError:
            st.error("Model not found. Run `python generate_demo_data.py` then `python train_model.py` first.")
            st.stop()

        try:
            y, sr = librosa.load(audio_data)
            duration = librosa.get_duration(y=y, sr=sr)
            if duration < 1.0:
                st.error("Recording too short. Please upload at least 1 second of audio.")
                st.stop()

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, y, sr)
                sound = parselmouth.Sound(tmp.name)
                tmp_path = tmp.name

            features = extract_features(sound, y, sr)
            os.unlink(tmp_path)

            # Predict
            feature_vector = np.array([[features.get(f, 0) for f in feature_names]])
            prediction = model.predict(feature_vector)[0]
            probability = model.predict_proba(feature_vector)[0]

            # ── BIG RESULT BANNER (full width, first thing the user sees) ──
            label = "Pathological Indicators Detected" if prediction == 1 else "Within Normal Range"
            confidence = probability[prediction] * 100
            if prediction == 1:
                banner_bg = "#fde8e8"
                banner_border = "#d32f2f"
                banner_color = "#b71c1c"
                icon = "⚠️"
                interpretation = ("The model detected patterns in your voice that are commonly "
                                  "associated with vocal cord irregularities, such as unsteady "
                                  "pitch or inconsistent volume.")
            else:
                banner_bg = "#e8f5e9"
                banner_border = "#2e7d32"
                banner_color = "#1b5e20"
                icon = "✅"
                interpretation = ("Your voice measurements fall within typical healthy ranges. "
                                  "The model did not detect patterns commonly linked to vocal "
                                  "cord problems.")

            st.markdown(f"""
            <div style="background:{banner_bg}; border-left:6px solid {banner_border};
                        border-radius:10px; padding:24px 28px; margin-bottom:20px;">
                <h1 style="color:{banner_color}; margin:0 0 6px 0; font-size:1.8rem;">
                    {icon} {label}
                </h1>
                <p style="color:#333; font-size:1.05rem; margin:0 0 10px 0;">
                    <em>{interpretation}</em>
                </p>
                <p style="margin:0; font-size:1rem;">
                    Confidence: <strong>{confidence:.1f}%</strong>
                    <span style="color:#777; margin-left:16px; font-size:0.85rem;">
                        P(Healthy) = {probability[0]:.2f} &nbsp;|&nbsp; P(Pathological) = {probability[1]:.2f}
                    </span>
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="disclaimer">
                <strong>This is a referral flag, not a diagnosis.</strong> This tool checks whether
                your voice shows patterns associated with vocal cord irregularity — including
                conditions like nodules, polyps, paralysis, or early laryngeal changes.  It does
                <strong>not</strong> identify specific conditions or replace a clinical examination.<br><br>
                <strong>Next step:</strong> If pathological indicators were detected, or if you
                have persistent hoarseness lasting more than 2–3 weeks, the recommended action is
                referral for <strong>laryngoscopy</strong> by an ENT (ear, nose &amp; throat) specialist.
                Early examination is especially important because voice changes can be an early sign
                of laryngeal cancer (American Cancer Society, 2024).
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div style="background:#e3f2fd; border-left:5px solid #1565c0;
                        border-radius:8px; padding:14px 18px; margin-top:12px;">
                <strong>⚗️ Trained on simulated data:</strong> This model was trained on
                synthetic voice measurements, <strong>not</strong> real clinical recordings.
                Results are for demonstration purposes only and should not be used for
                medical decisions.
            </div>
            """, unsafe_allow_html=True)

            # ── Waveform (full width, below the result) ──
            st.markdown("---")
            st.subheader("Waveform")
            fig, ax = plt.subplots(figsize=(10, 3))
            librosa.display.waveshow(y, sr=sr, color="#007bff", ax=ax)
            ax.set_facecolor("#f8f9fa")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            st.pyplot(fig)
            plt.close()

            st.markdown("---")
            st.subheader("Key Acoustic Biomarkers")
            st.caption("These measurements describe different qualities of your voice. Each is compared against a typical healthy average.")

            reference = {
                "jitter_local": ("Pitch Stability (Jitter)", "How steady your pitch is from cycle to cycle — higher values suggest irregular vibration", 0.005, "%", 100),
                "shimmer_local": ("Volume Consistency (Shimmer)", "How evenly your voice volume holds — higher values suggest the vocal cords aren't closing evenly", 0.028, "%", 100),
                "hnr": ("Voice Clarity (HNR)", "Ratio of clear tone to noise in your voice — higher is clearer and healthier", 21.5, "dB", 1),
                "f0_mean": ("Base Pitch (F0)", "Your fundamental speaking frequency — varies by age and sex", 178, "Hz", 1),
            }
            bio_cols = st.columns(4)
            for i, (key, (name, description, ref_val, unit, mult)) in enumerate(reference.items()):
                val = features.get(key, 0) * mult
                ref_display = ref_val * mult
                with bio_cols[i]:
                    st.metric(label=name, value=f"{val:.3f} {unit}",
                              delta=f"{val - ref_display:+.3f} vs healthy avg",
                              delta_color="inverse" if key != "hnr" else "normal")
                    st.caption(description)

            st.markdown("---")
            st.subheader("MFCC Spectrogram")
            st.caption("MFCCs capture the shape of your vocal tract over time — think of them as a fingerprint of how your voice sounds, independent of what you're saying.")
            mfccs_full = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            fig_heat = go.Figure(data=go.Heatmap(
                z=mfccs_full, colorscale="Blues",
                y=[f"MFCC {i+1}" for i in range(13)],
            ))
            fig_heat.update_layout(height=300, margin=dict(l=0, r=0, b=30, t=10),
                                   xaxis_title="Time Frame", yaxis_title="Coefficient")
            st.plotly_chart(fig_heat, use_container_width=True)

            # Radar chart
            st.subheader("Feature Profile vs. Healthy Baseline")
            st.caption("This radar chart shows how your voice compares to a typical healthy voice across five key measures. If your shape closely matches the green 'Healthy Baseline', your voice is within normal ranges.")
            radar_keys = ["jitter_local", "shimmer_local", "hnr", "spectral_flatness", "zcr"]
            radar_labels = ["Jitter", "Shimmer", "HNR", "Spec. Flatness", "ZCR"]
            healthy_bl = {"jitter_local": 0.005, "shimmer_local": 0.028, "hnr": 21.5,
                          "spectral_flatness": 0.025, "zcr": 0.052}
            path_max = {"jitter_local": 0.03, "shimmer_local": 0.12, "hnr": 30.0,
                        "spectral_flatness": 0.15, "zcr": 0.15}

            user_norm = [min(features.get(f, 0) / path_max[f], 1.0) for f in radar_keys]
            base_norm = [healthy_bl[f] / path_max[f] for f in radar_keys]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=base_norm + [base_norm[0]], theta=radar_labels + [radar_labels[0]],
                fill="toself", name="Healthy Baseline", line_color="#2e7d32"))
            fig_radar.add_trace(go.Scatterpolar(
                r=user_norm + [user_norm[0]], theta=radar_labels + [radar_labels[0]],
                fill="toself", name="Your Sample", line_color="#007bff"))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                                     height=400, margin=dict(l=40, r=40, b=40, t=40))
            st.plotly_chart(fig_radar, use_container_width=True)

            with st.expander("View All Extracted Features"):
                feat_df = pd.DataFrame([features]).T
                feat_df.columns = ["Value"]
                feat_df.index.name = "Feature"
                st.dataframe(feat_df)

            # --- Download Report ---
            st.markdown("---")
            report_html = generate_report_html(features, prediction, probability, duration)
            st.download_button(
                label="📄 Download Analysis Report (HTML)",
                data=report_html,
                file_name=f"vocalpath_report_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
                mime="text/html",
            )

        except Exception as e:
            st.error(f"Error processing audio: {e}")
            st.info("Ensure the file is a valid audio recording (.wav, .mp3, .ogg, .flac, .m4a) of a sustained vowel.")

    else:
        st.markdown("""
        <div class="mobile-sidebar-hint">
            <div class="hint-icon">🎙️</div>
            <strong>Tap the <code>&gt;</code> arrow in the top-left corner</strong> to open the sidebar,
            where you can <strong>upload an audio file</strong> or <strong>record your voice</strong>
            directly from your phone.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        ### Why Voice Analysis?

        Your voice is produced by your **vocal cords** — two small folds of tissue in your throat that vibrate
        hundreds of times per second when you speak. When these vocal cords are healthy, they vibrate
        symmetrically and predictably. When something is wrong — nodules, polyps, inflammation, or
        neurological conditions — those vibrations become **irregular** in ways that are measurable but
        often too subtle for the human ear to notice.

        **This matters for cancer screening.** Laryngeal cancer — cancer of the voice box — is one of the
        most common head-and-neck cancers, and **persistent hoarseness** is often its earliest symptom
        (American Cancer Society, 2024). Yet many patients dismiss voice changes or wait months before
        seeing a doctor. An automated voice screening tool could flag at-risk individuals earlier and
        prompt timely referral for laryngoscopy — the gold-standard diagnostic examination.

        **Think of it like a blood test, but for your voice.** Instead of measuring cholesterol or blood sugar,
        we measure acoustic properties like pitch stability, volume consistency, and voice clarity. Machine
        learning can then look at all of these measurements together to flag voices that show patterns
        associated with vocal cord problems — serving as a **referral trigger** rather than a diagnosis.

        ---

        ### How It Works
        1. **Upload** a recording of a sustained vowel sound ('Ah')
        2. **Feature Extraction** — We measure 26 properties of your voice, including pitch stability, volume consistency, and vocal tract shape
        3. **Classification** — A trained machine learning model checks whether those measurements match healthy or pathological patterns
        4. **Visualization** — Interactive charts compare your voice against healthy baselines so you can see the differences yourself
        """)

# --- METHODOLOGY TAB ---
with tab_methodology:
    st.header("Methodology")

    st.markdown("""
    ### The Big Picture

    When you speak, your vocal cords vibrate to produce sound. This tool records those vibrations
    and measures **26 different properties** of your voice — things like how steady your pitch is,
    how consistent your volume is, and how much background noise is in your voice signal.

    We then feed all 26 measurements into a machine learning model that has been trained on hundreds
    of voice samples (both healthy and pathological) to learn which patterns are associated with
    vocal cord problems. The model outputs a prediction — "healthy" or "pathological" — along with
    a confidence score.

    **In short:** Record your voice → measure 26 properties → compare against known patterns → get a result.

    ---
    """)

    st.markdown("### What We Measure")

    st.markdown("""
    The 26 features fall into three groups:

    **Voice Stability Measures** — These come from a tool called [Praat](https://www.fon.hum.uva.nl/praat/),
    widely used in speech science. They measure how regular your vocal cord vibrations are:
    - **Jitter** — Is your pitch steady from one vibration cycle to the next? (Higher = less stable)
    - **Shimmer** — Is your volume steady from one cycle to the next? (Higher = less consistent)
    - **HNR (Harmonics-to-Noise Ratio)** — How much of your voice is clear tone vs. noise? (Higher = clearer)
    - **F0 (Fundamental Frequency)** — Your base speaking pitch, in Hz

    **Vocal Tract Shape (MFCCs 1–13)** — These capture the shape of your mouth and throat as you speak,
    like a fingerprint of your voice. Extracted using a library called [Librosa](https://librosa.org/).

    **Spectral & Energy Features** — These describe the overall "texture" of your voice:
    - *Spectral centroid* (brightness), *bandwidth* (spread), *flatness* (noise-like vs. tonal), *rolloff* (high-frequency content)
    - *Zero-crossing rate* and *RMS energy* (loudness)
    """)

    with st.expander("Technical Details: Feature Extraction"):
        st.markdown("""
        #### Perturbation Measures (Praat/Parselmouth)
        | Feature | Description | Clinical Significance |
        |---|---|---|
        | **Jitter (local)** | Cycle-to-cycle pitch period variation | Elevated in vocal fold lesions, nodules, polyps |
        | **Jitter (RAP)** | Relative average perturbation | Smoothed jitter, less noise-sensitive |
        | **Shimmer (local)** | Cycle-to-cycle amplitude variation | Elevated in laryngeal pathology |
        | **Shimmer (APQ3)** | 3-point amplitude perturbation quotient | Smoothed shimmer |
        | **HNR** | Harmonics-to-noise ratio (dB) | Low = breathy/hoarse voice |

        #### Spectral Features (Librosa)
        | Feature | Description |
        |---|---|
        | **MFCCs (1-13)** | Vocal tract shape encoding |
        | **Spectral Centroid** | Perceived brightness |
        | **Spectral Bandwidth** | Spectral spread |
        | **Spectral Flatness** | Noise-like vs. tone-like |
        | **Spectral Rolloff** | High-frequency energy boundary |

        #### Temporal Features
        | Feature | Description |
        |---|---|
        | **ZCR** | Zero-crossing rate |
        | **RMS** | Root mean square energy |
        | **F0 (mean, std)** | Fundamental frequency and variability |
        """)

    st.markdown("---")
    st.markdown("### How the Model Learns")

    st.markdown("""
    We trained **three different machine learning models** on the same dataset and picked the one
    that performed best:

    - **Random Forest** — Makes predictions by averaging the votes of 100 decision trees. Good at handling complex feature interactions.
    - **SVM (Support Vector Machine)** — Finds the optimal boundary between healthy and pathological voices in the feature space.
    - **Logistic Regression** — A simpler model that estimates the probability of each class directly.

    Each model was tested using **5-fold cross-validation**: the data is split into 5 parts, and the
    model is trained on 4 parts and tested on the 1 remaining part, rotating through all 5 combinations.
    This ensures the reported accuracy reflects how the model performs on data it hasn't seen before.

    The model with the highest **F1 score** (a balance of precision and recall) is automatically selected.
    """)

    with st.expander("Technical Details: Model Configuration"):
        st.markdown("""
        - **Random Forest:** 100 estimators, max depth 10, balanced class weights
        - **SVM (RBF kernel):** Balanced class weights, probability calibration enabled
        - **Logistic Regression:** Balanced class weights, default regularisation
        - **Preprocessing:** StandardScaler (zero mean, unit variance)
        - **Selection criterion:** Highest macro-averaged F1 score across 5-fold stratified CV
        """)

    st.markdown("---")
    st.markdown("### Clinical Validity & Limitations")
    st.markdown("""
    #### Technical Validity (what we *can* show)
    - The best model achieves ~88 % accuracy on held-out synthetic data via 5-fold stratified cross-validation
    - Performance metrics (precision, recall, F1, AUC) are reported transparently in the **"How Accurate Is This Tool?"** tab
    - Feature extraction relies on Praat and Librosa — the same libraries used in published voice-pathology research

    #### Clinical Validity Gaps (what is still missing)
    - **No real-patient validation** — the model has only seen synthetic samples, not recordings from diagnosed patients
    - **No external dataset testing** — we have not evaluated on standard clinical corpora such as the Saarbrücken Voice Database (SVD) or MEEI
    - **No demographic sub-group analysis** — performance may vary by age, sex, accent, or language
    - **No prospective study** — the tool has not been used alongside ENT specialists to measure real-world added value
    - **Recording conditions uncontrolled** — consumer microphones, background noise, and room acoustics introduce variability that the model was not trained on

    #### What a Full Clinical Validation Would Look Like
    If this prototype were to move toward real use, the following steps would be needed:
    1. **Ethical approval** from an Institutional Review Board (IRB) for collecting patient voice samples
    2. **Gold-standard labels** from ENT specialists using laryngoscopy, following the STARD reporting guideline for diagnostic accuracy studies
    3. **External validation** on at least one independent clinical dataset (e.g., SVD, Hospital Príncipe de Asturias corpus)
    4. **Sub-group analysis** to check for bias across age, sex, and language
    5. **Transparent reporting** following the TRIPOD checklist for prediction-model development and validation
    """)

    st.markdown("---")
    st.markdown("### Benefits & Risks of AI in Voice Screening")
    st.markdown("""
    #### Potential Benefits
    - **Early detection** — voice changes can appear before other symptoms; automated screening could catch problems sooner
    - **Accessibility** — a smartphone-based tool could reach people in under-served or remote communities without nearby ENT clinics
    - **Reduced clinician workload** — pre-screening could help prioritise referrals, saving specialists' time for complex cases
    - **Objective measurement** — acoustic features provide quantitative data that complement subjective perceptual evaluations
    - **Patient empowerment** — individuals can self-monitor longitudinally and share results with their healthcare provider

    #### Potential Risks
    - **False negatives** — a missed pathology could delay treatment if users treat a "Healthy" result as a clean bill of health
    - **False positives** — unnecessary anxiety and potentially unnecessary specialist visits
    - **Over-reliance** — users (or even clinicians) might trust the tool more than warranted given its current validation level
    - **Data privacy** — voice recordings are biometric data; improper storage or transmission could violate privacy regulations (e.g., GDPR, HIPAA)
    - **Bias and fairness** — if training data does not represent all demographic groups, the tool may perform worse for under-represented populations
    - **Regulatory gap** — deploying an unvalidated screening tool without CE/FDA clearance could pose safety and legal risks

    #### Ethical Safeguards in This Prototype
    - Clear **"research prototype" disclaimers** are shown before every result
    - The tool does **not** store or transmit user recordings
    - A **limitation disclaimer** warns that results are not a substitute for professional diagnosis
    - All model performance data is **openly reported** so users can judge reliability for themselves
    """)

    st.markdown("---")
    st.markdown("### Spectrogram-Based Deep Learning (Experimental)")
    st.markdown("""
    In addition to the tabular pipeline above, this project includes an **experimental CNN
    (Convolutional Neural Network)** approach in `train_spectrogram_cnn.py` that classifies
    voice pathology directly from **mel-spectrograms** — 2D heatmap images of the voice's
    frequency content over time.

    **Why two approaches?**
    - The **tabular pipeline** (Random Forest / SVM / Logistic Regression on 26 features) is
      more *interpretable* — clinicians can see exactly which measurements drove the decision
    - The **spectrogram CNN** is more *expressive* — it preserves temporal patterns that
      averaging destroys, such as pitch instability that worsens over the course of a sustained
      vowel

    This dual-pipeline design mirrors real-world clinical AI development, where interpretable
    models are preferred for regulatory approval, but deep learning models often achieve higher
    raw accuracy on complex signals.

    The CNN architecture: 3 convolutional blocks (Conv2D → BatchNorm → MaxPool) followed by
    a dense classification head with dropout regularisation.  It is trained on synthetic
    mel-spectrograms and serves as a proof-of-concept for spectrogram-based voice analysis.
    """)

    st.markdown("---")
    st.markdown("### Future Directions")
    st.markdown("""
    This prototype demonstrates a feasible end-to-end pipeline.  To move toward clinical
    utility, the following extensions would be valuable:

    - **Self-supervised speech models** — Fine-tuning foundation models such as
      [wav2vec 2.0](https://arxiv.org/abs/2006.11477) (Facebook/Meta) or
      [HuBERT](https://arxiv.org/abs/2106.07447) on clinical voice data.  These models
      learn rich audio representations from unlabelled speech and have shown state-of-the-art
      results in speech classification tasks including pathology detection.
    - **Whisper embeddings** — OpenAI's [Whisper](https://arxiv.org/abs/2212.04356) model,
      originally designed for speech recognition, produces intermediate representations that
      encode voice quality information useful for screening tasks.
    - **Real clinical datasets** — Validation on the Saarbrücken Voice Database (SVD),
      MEEI corpus, or hospital-collected samples with laryngoscopy-confirmed labels.
    - **Longitudinal monitoring** — Tracking voice changes over weeks/months to detect
      gradual deterioration, which is particularly relevant for early cancer detection.
    - **mHealth integration** — Deploying as a smartphone app that records periodic voice
      samples and alerts users if a concerning trend is detected, aligning with mobile health
      (mHealth) principles for accessible screening.
    """)

    st.markdown("---")
    st.markdown("### References")
    st.markdown("""
    1. Teixeira, J.P. et al. (2013). "Vocal Acoustic Analysis — Jitter, Shimmer and HNR Parameters." *Procedia Technology*, 9.
    2. Godino-Llorente, J.I. et al. (2006). "Dimensionality Reduction of a Pathological Voice Quality Feature Set." *IEEE Trans. BME*, 53(10).
    3. Martinez, D. et al. (2012). "Voice Pathology Detection on the Saarbrucken Voice Database."
    4. Boersma, P. & Weenink, D. (2023). Praat: doing phonetics by computer.
    5. American Cancer Society (2024). "Signs and Symptoms of Laryngeal and Hypopharyngeal Cancers."
    6. Baevski, A. et al. (2020). "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations." *NeurIPS*.
    7. Hsu, W.-N. et al. (2021). "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units." *IEEE/ACM Trans. ASLP*.
    8. Radford, A. et al. (2022). "Robust Speech Recognition via Large-Scale Weak Supervision." *arXiv:2212.04356*.
    9. Hemmerling, D. et al. (2016). "Voice data mining for laryngeal pathology assessment." *Computers in Biology and Medicine*, 69.
    """)

# --- HOW ACCURATE IS THIS TOOL? TAB ---
with tab_model:
    st.header("How Accurate Is This Tool?")
    st.markdown("""
    This section is **not** about your personal voice — it's about the
    **system's overall reliability**.  Before trusting any tool, you should
    know how often it gets things right and where it struggles.  Below we
    show exactly that: which algorithm we chose, how each one scored, and
    what kinds of mistakes the model makes.
    """)
    try:
        _, _, metrics, importance = load_model()
        cm, roc_data, pr_data, lc_data = load_eval_artifacts()

        # --- Model Comparison Table ---
        if "models" in metrics:
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            st.markdown("### Which Algorithm Won?")
            st.markdown("""
            We tested three common machine-learning algorithms and picked the one
            that was best at correctly identifying **both** healthy and pathological
            voices.  Each was tested using **5-fold cross-validation** — the data is
            split into 5 chunks, and the model is trained on 4 while being tested on
            the 1 it hasn't seen, rotating through all 5.  This prevents the model
            from just memorising the answers.
            """)
            # Tooltip descriptions for each algorithm
            model_tooltips = {
                "Random Forest": (
                    "Builds many decision trees (like a panel of doctors), each looking "
                    "at a random subset of voice measurements. They vote together to "
                    "make the final call — the majority wins."
                ),
                "SVM (RBF)": (
                    "Support Vector Machine — draws a boundary between healthy and "
                    "pathological voices in a high-dimensional space. The RBF kernel "
                    "lets that boundary curve, so it can capture complex patterns."
                ),
                "Logistic Regression": (
                    "The simplest approach: adds up all the voice measurements "
                    "(each multiplied by a learned weight) and converts the total "
                    "into a probability between 0% and 100%. Fast and easy to interpret."
                ),
            }
            tooltip_css = """
            <style>
            .model-tooltip-container { display: inline-block; position: relative; }
            .model-tooltip-icon {
                display: inline-flex; align-items: center; justify-content: center;
                width: 18px; height: 18px; border-radius: 50%;
                background: #e8f0fe; color: #1e3a8a; font-size: 12px;
                font-weight: bold; cursor: help; margin-left: 5px;
                vertical-align: middle; border: 1px solid #c4d6f0;
            }
            .model-tooltip-icon:hover + .model-tooltip-text,
            .model-tooltip-text:hover {
                visibility: visible; opacity: 1;
            }
            .model-tooltip-text {
                visibility: hidden; opacity: 0;
                width: 280px; background: #1e293b; color: #f1f5f9;
                text-align: left; border-radius: 8px; padding: 10px 14px;
                position: absolute; z-index: 1000;
                bottom: 130%; left: 50%; transform: translateX(-50%);
                font-size: 0.82rem; line-height: 1.45;
                box-shadow: 0 4px 12px rgba(0,0,0,0.25);
                transition: opacity 0.2s;
            }
            .model-tooltip-text::after {
                content: ""; position: absolute;
                top: 100%; left: 50%; margin-left: -6px;
                border-width: 6px; border-style: solid;
                border-color: #1e293b transparent transparent transparent;
            }
            .comp-table { width: 100%; border-collapse: collapse; margin: 8px 0 12px 0; }
            .comp-table th {
                background: #f8fafc; color: #475569; font-size: 0.85rem;
                padding: 10px 14px; text-align: left; border-bottom: 2px solid #e2e8f0;
            }
            .comp-table td {
                padding: 10px 14px; font-size: 0.9rem; border-bottom: 1px solid #f1f5f9;
            }
            .comp-table tr:hover td { background: #f8fafc; }
            </style>
            """
            st.markdown(tooltip_css, unsafe_allow_html=True)

            comparison_data = []
            for model_name, m in metrics["models"].items():
                comparison_data.append({
                    "Model": model_name,
                    "Accuracy": m["accuracy"],
                    "Precision": m["precision"],
                    "Recall": m["recall"],
                    "F1 Score": m["f1_score"],
                    "AUC-ROC": m["auc_roc"],
                })

            # Build HTML table with tooltip icons inside each model-name cell
            metric_cols = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"]
            table_html = "<table class='comp-table'><thead><tr><th>Model</th>"
            for c in metric_cols:
                table_html += f"<th>{c}</th>"
            table_html += "</tr></thead><tbody>"
            for row in comparison_data:
                name = row["Model"]
                tip = model_tooltips.get(name, "")
                name_cell = (
                    f'<span class="model-tooltip-container">{name}'
                    f'<span class="model-tooltip-icon">?</span>'
                    f'<span class="model-tooltip-text">{tip}</span></span>'
                    if tip else name
                )
                table_html += f"<tr><td>{name_cell}</td>"
                for c in metric_cols:
                    table_html += f"<td>{row[c]}</td>"
                table_html += "</tr>"
            table_html += "</tbody></table>"
            st.markdown(table_html, unsafe_allow_html=True)

            best = metrics.get("best_model", "N/A")
            st.success(f"**Selected model:** {best} (highest F1 score)")

        st.markdown("<br><br><br>", unsafe_allow_html=True)

        # --- Top-level metrics for best model ---
        st.markdown("### Best Model — Score Card")
        st.markdown("""
        Each number below answers a slightly different question about model quality.
        All are between 0 and 1 — **closer to 1 is better**.
        """)

        metric_explanations = {
            "Accuracy": "Of all voices tested, how many did the model label correctly?",
            "Precision": "When the model says 'pathological', how often is it right?",
            "Recall": "Of all truly pathological voices, how many did the model catch?",
            "F1 Score": "A single number that balances Precision and Recall.",
            "AUC-ROC": "Overall ability to tell healthy from pathological (see curve below).",
        }
        mcols = st.columns(5)
        ci = metrics.get("bootstrap_ci", {})
        for col, (name, key) in zip(mcols, [
            ("Accuracy", "accuracy"), ("Precision", "precision"),
            ("Recall", "recall"), ("F1 Score", "f1_score"), ("AUC-ROC", "auc_roc"),
        ]):
            with col:
                ci_key = key.replace("_score", "").replace("auc_roc", "roc_auc")
                ci_info = ci.get(ci_key, {})
                ci_text = ""
                if ci_info:
                    ci_text = f"<p style='color:#888;font-size:0.72rem;margin:2px 0 0 0'>95% CI: [{ci_info.get('ci_low','')}, {ci_info.get('ci_high','')}]</p>"
                st.markdown(f"""
                <div class="metric-card" style="text-align:center">
                    <p style='color:gray;margin:0'>{name}</p>
                    <h2 style='color:#1e3a8a;margin:5px 0'>{metrics.get(key, 'N/A')}</h2>
                    <p style='color:#555;font-size:0.75rem;margin:4px 0 0 0'>{metric_explanations[name]}</p>
                    {ci_text}
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br><br><br>", unsafe_allow_html=True)

        # --- Confusion Matrix and ROC side by side ---
        cm_col, roc_col = st.columns(2)

        with cm_col:
            st.markdown("### Confusion Matrix")
            st.caption(
                "A confusion matrix shows **where the model gets it right and where it "
                "gets confused**. Read each row as: 'Out of all voices that were *actually* "
                "this class, how did the model label them?' The diagonal (top-left and "
                "bottom-right) shows correct predictions; off-diagonal cells are mistakes."
            )
            labels = ["Healthy", "Pathological"]
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm.values, x=labels, y=labels,
                colorscale=[[0, "#f0f4ff"], [1, "#1e3a8a"]],
                text=cm.values, texttemplate="%{text}",
                textfont={"size": 20}, showscale=False,
            ))
            fig_cm.update_layout(
                xaxis_title="What the model predicted",
                yaxis_title="What the voice actually was",
                height=350, margin=dict(l=0, r=0, b=40, t=10),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_cm, use_container_width=True, config={'displayModeBar': False})

        with roc_col:
            st.markdown("### ROC Curve")
            st.caption(
                "The ROC curve shows the trade-off between **catching sick voices** "
                "(vertical axis) and **accidentally flagging healthy ones** (horizontal "
                "axis). The blue curve hugging the top-left corner means the model is "
                "good at telling the two apart. The grey dashed line is what random "
                "guessing would look like. The **AUC** number is the area under the blue "
                "curve — 1.0 = perfect, 0.5 = random coin flip."
            )
            roc_auc_val = np.trapezoid(roc_data["tpr"], roc_data["fpr"])
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=roc_data["fpr"], y=roc_data["tpr"],
                mode="lines", name=f"AUC = {roc_auc_val:.3f}",
                line=dict(color="#007bff", width=2),
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Random guessing",
                line=dict(color="gray", dash="dash"),
            ))
            fig_roc.update_layout(
                xaxis_title="False alarm rate (healthy voices wrongly flagged)",
                yaxis_title="Detection rate (sick voices correctly caught)",
                height=350, margin=dict(l=0, r=0, b=40, t=10),
                legend=dict(x=0.55, y=0.1),
            )
            st.plotly_chart(fig_roc, use_container_width=True, config={'displayModeBar': False})

        # --- Precision-Recall Curve ---
        if pr_data is not None:
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            st.markdown("### Precision-Recall Curve")
            st.markdown("""
            **Why this matters for cancer screening:** In a screening tool, missing a
            sick person (**false negative**) is far worse than a false alarm. A missed
            case could delay cancer diagnosis, while a false alarm just means an
            unnecessary laryngoscopy referral. So we want **high recall** (catch as
            many pathological voices as possible), even if precision drops a little.

            The curve below shows this tradeoff. The **Average Precision (AP)** score
            summarises overall performance — closer to 1.0 is better.
            """)
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(
                x=pr_data["recall"], y=pr_data["precision"],
                mode="lines",
                name=f"AP = {metrics.get('average_precision', 'N/A')}",
                line=dict(color="#2563eb", width=2),
                fill="tozeroy",
                fillcolor="rgba(37,99,235,0.08)",
            ))
            baseline = 0.51  # approximate class ratio
            fig_pr.add_trace(go.Scatter(
                x=[0, 1], y=[baseline, baseline], mode="lines",
                name=f"Random baseline ({baseline:.2f})",
                line=dict(color="gray", dash="dash"),
            ))
            fig_pr.update_layout(
                xaxis_title="Recall (how many sick voices we catch)",
                yaxis_title="Precision (how often 'sick' predictions are correct)",
                height=380, margin=dict(l=0, r=0, b=40, t=10),
                legend=dict(x=0.02, y=0.08),
                xaxis=dict(range=[0, 1.02]),
                yaxis=dict(range=[0, 1.02]),
            )
            st.plotly_chart(fig_pr, use_container_width=True, config={'displayModeBar': False})

        # --- Bootstrap Confidence Intervals ---
        if "bootstrap_ci" in metrics:
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            st.markdown("### Confidence Intervals (Bootstrap)")
            st.markdown("""
            A single accuracy number can be misleading. **How confident are we?**
            We resampled the model's predictions 1,000 times to compute
            **95% confidence intervals** — the range within which the true
            performance very likely falls.
            """)
            ci = metrics["bootstrap_ci"]
            ci_cols = st.columns(5)
            ci_map = [
                ("Accuracy", "accuracy"), ("F1 Score", "f1"),
                ("Recall", "recall"), ("Precision", "precision"),
                ("AUC-ROC", "roc_auc"),
            ]
            for col, (label, key) in zip(ci_cols, ci_map):
                with col:
                    d = ci.get(key, {})
                    mean_val = d.get("mean", "N/A")
                    ci_low = d.get("ci_low", "N/A")
                    ci_high = d.get("ci_high", "N/A")
                    st.markdown(f"""
                    <div class="metric-card" style="text-align:center">
                        <p style='color:gray;margin:0'>{label}</p>
                        <h2 style='color:#1e3a8a;margin:5px 0'>{mean_val}</h2>
                        <p style='color:#555;font-size:0.8rem;margin:4px 0 0 0'>95% CI: [{ci_low}, {ci_high}]</p>
                    </div>
                    """, unsafe_allow_html=True)

        # --- Learning Curve ---
        if lc_data is not None:
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            st.markdown("### Learning Curve — Would More Data Help?")
            st.markdown("""
            This chart shows how the model's performance (F1 score) changes as we
            give it more training data. If the blue and red lines have converged,
            adding more data won't help much. If there's still a big gap, the model
            could improve with a larger dataset.
            """)
            fig_lc = go.Figure()
            fig_lc.add_trace(go.Scatter(
                x=lc_data["train_size"], y=lc_data["train_mean"],
                mode="lines+markers", name="Training F1",
                line=dict(color="#2563eb", width=2),
            ))
            fig_lc.add_trace(go.Scatter(
                x=lc_data["train_size"], y=lc_data["val_mean"],
                mode="lines+markers", name="Validation F1",
                line=dict(color="#d32f2f", width=2),
            ))
            # Add confidence bands
            fig_lc.add_trace(go.Scatter(
                x=list(lc_data["train_size"]) + list(lc_data["train_size"][::-1]),
                y=list(lc_data["val_mean"] + lc_data["val_std"]) +
                  list((lc_data["val_mean"] - lc_data["val_std"])[::-1]),
                fill="toself", fillcolor="rgba(211,47,47,0.1)",
                line=dict(width=0), showlegend=False,
            ))
            fig_lc.update_layout(
                xaxis_title="Number of Training Samples",
                yaxis_title="F1 Score",
                height=400, margin=dict(l=0, r=0, b=40, t=10),
                legend=dict(x=0.6, y=0.15),
                yaxis=dict(range=[0.5, 1.02]),
            )
            st.plotly_chart(fig_lc, use_container_width=True, config={'displayModeBar': False})

        st.markdown("<br><br><br>", unsafe_allow_html=True)

        # --- Feature importance ---
        st.markdown("### Which Voice Measurements Matter Most?")
        st.markdown("""
        The chart below ranks the 15 most important voice measurements the model
        relies on.  Longer bars = the model pays more attention to that measurement
        when deciding whether a voice is healthy or pathological.
        """)
        top = importance.head(15).copy()

        # Replace technical parameter names with plain-English labels
        friendly_feature_names = {
            "f0_mean": "Base Pitch",
            "f0_std": "Pitch Wobble",
            "jitter_local": "Pitch Instability (Jitter)",
            "jitter_rap": "Rapid Pitch Variation (Jitter RAP)",
            "shimmer_local": "Volume Instability (Shimmer)",
            "shimmer_apq3": "Volume Variation (Shimmer APQ3)",
            "hnr": "Voice Clarity (HNR)",
            "mfcc_1": "Vocal Tract Shape 1 (MFCC-1)",
            "mfcc_2": "Vocal Tract Shape 2 (MFCC-2)",
            "mfcc_3": "Vocal Tract Shape 3 (MFCC-3)",
            "mfcc_4": "Vocal Tract Shape 4 (MFCC-4)",
            "mfcc_5": "Vocal Tract Shape 5 (MFCC-5)",
            "mfcc_6": "Vocal Tract Shape 6 (MFCC-6)",
            "mfcc_7": "Vocal Tract Shape 7 (MFCC-7)",
            "mfcc_8": "Vocal Tract Shape 8 (MFCC-8)",
            "mfcc_9": "Vocal Tract Shape 9 (MFCC-9)",
            "mfcc_10": "Vocal Tract Shape 10 (MFCC-10)",
            "mfcc_11": "Vocal Tract Shape 11 (MFCC-11)",
            "mfcc_12": "Vocal Tract Shape 12 (MFCC-12)",
            "mfcc_13": "Vocal Tract Shape 13 (MFCC-13)",
            "spectral_centroid": "Voice Brightness",
            "spectral_bandwidth": "Frequency Spread",
            "spectral_flatness": "Noisiness (Spectral Flatness)",
            "spectral_rolloff": "High-Frequency Energy",
            "zcr": "Zero-Crossing Rate",
            "rms": "Loudness (RMS Energy)",
        }
        top["feature"] = top["feature"].map(lambda x: friendly_feature_names.get(x, x))

        fig_imp = px.bar(top, x="importance", y="feature", orientation="h",
                         color="importance", color_continuous_scale="Blues")
        fig_imp.update_layout(height=400, margin=dict(l=0, r=0, b=0, t=10),
                              yaxis=dict(autorange="reversed"), showlegend=False)
        st.plotly_chart(fig_imp, use_container_width=True, config={'displayModeBar': False})

        st.markdown(f"""
        **Training Details:** The model was trained on {metrics.get('n_samples', 'N/A')} synthetic voice samples,
        each described by {metrics.get('n_features', 'N/A')} measurements, and validated with
        {metrics.get('cv_folds', 'N/A')}-fold cross-validation (the data is split into {metrics.get('cv_folds', 'N/A')} parts;
        the model trains on 4 and tests on the 1 it hasn't seen, rotating through all parts).
        *{metrics.get('note', '')}*
        """)

        # --- CNN Comparison ---
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown("### Deep Learning Alternative: Spectrogram CNN")
        st.markdown("""
        The tabular pipeline above works on **26 averaged numbers** per recording.
        We also built a **Convolutional Neural Network (CNN)** that classifies
        voice pathology directly from **mel-spectrogram images** — 2D heatmaps of
        frequency content over time.  This preserves temporal patterns that
        averaging destroys.
        """)

        try:
            with open("model/cnn_metrics.json") as f:
                cnn_metrics = json.load(f)

            cnn_col1, cnn_col2, cnn_col3 = st.columns(3)
            with cnn_col1:
                st.markdown(f"""
                <div class="metric-card" style="text-align:center">
                    <p style='color:gray;margin:0'>CNN Accuracy</p>
                    <h2 style='color:#1e3a8a;margin:5px 0'>{cnn_metrics.get('accuracy', 'N/A')}</h2>
                    <p style='color:#555;font-size:0.75rem;margin:4px 0 0 0'>On held-out test set</p>
                </div>
                """, unsafe_allow_html=True)
            with cnn_col2:
                st.markdown(f"""
                <div class="metric-card" style="text-align:center">
                    <p style='color:gray;margin:0'>CNN AUC-ROC</p>
                    <h2 style='color:#1e3a8a;margin:5px 0'>{cnn_metrics.get('auc_roc', 'N/A')}</h2>
                    <p style='color:#555;font-size:0.75rem;margin:4px 0 0 0'>Perfect separation = 1.0</p>
                </div>
                """, unsafe_allow_html=True)
            with cnn_col3:
                spec_shape = cnn_metrics.get('spectrogram_shape', [])
                shape_str = f"{spec_shape[0]}×{spec_shape[1]}" if len(spec_shape) >= 2 else "N/A"
                st.markdown(f"""
                <div class="metric-card" style="text-align:center">
                    <p style='color:gray;margin:0'>Input Size</p>
                    <h2 style='color:#1e3a8a;margin:5px 0'>{shape_str}</h2>
                    <p style='color:#555;font-size:0.75rem;margin:4px 0 0 0'>Mel bands × time frames</p>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("CNN Architecture & Training Details"):
                st.markdown(f"""
                | Property | Value |
                |---|---|
                | **Architecture** | {cnn_metrics.get('architecture', 'N/A')} |
                | **Training samples** | {cnn_metrics.get('n_train', 'N/A')} |
                | **Test samples** | {cnn_metrics.get('n_test', 'N/A')} |
                | **Epochs trained** | {cnn_metrics.get('epochs', 'N/A')} |
                | **Optimal threshold** | {cnn_metrics.get('optimal_threshold', 'N/A')} |

                **Why two pipelines?**
                - The **tabular model** (used in the Analysis tab) is more *interpretable* —
                  you can see exactly which 26 measurements drove the decision
                - The **spectrogram CNN** is more *expressive* — it learns directly from the
                  raw audio's frequency-time representation, capturing patterns like pitch
                  instability that worsens over the course of a sustained vowel

                The app uses the tabular pipeline for clinical interpretability.  The CNN
                demonstrates that spectrogram-based deep learning is a viable alternative
                approach for voice pathology detection.

                *{cnn_metrics.get('note', '')}*
                """)
        except FileNotFoundError:
            st.info("CNN not yet trained. Run `python train_spectrogram_cnn.py` to see results here.")
    except FileNotFoundError:
        st.warning("Model not yet trained. Run `python generate_demo_data.py` then `python train_model.py`.")

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Student Performance Predictor
# Developed by Ajim Sikandar Mujawar

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

st.set_page_config(
    page_title="Student Performance Predictor | Ajim Sikandar Mujawar",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: linear-gradient(135deg, #f7f9ff 0%, #f3f6fc 55%, #eef3ff 100%); }
    .block-container { padding-top: 1.8rem; padding-bottom: 2.5rem; max-width: 1180px; }
    .hero {
        padding: 2rem 2.1rem; border-radius: 24px;
        background: linear-gradient(120deg, #14213d 0%, #233f77 58%, #3566b8 100%);
        color: white; box-shadow: 0 16px 40px rgba(31, 55, 103, .16);
        margin-bottom: 1.2rem;
    }
    .hero-kicker { text-transform: uppercase; letter-spacing: .16em; font-size: .75rem; font-weight: 700; color: #b8d2ff; }
    .hero h1 { font-family: 'Manrope', sans-serif; font-size: clamp(2rem, 4vw, 3rem); line-height: 1.12; margin: .55rem 0 .7rem 0; color: #fff; }
    .hero p { color: #e2eaff; font-size: 1.02rem; max-width: 760px; margin-bottom: .65rem; }
    .hero-byline { font-size: .84rem; color: #c7d8fa; margin-top: 1.15rem; }
    .section-card {
        background: #fff; border: 1px solid #e3eaf5; border-radius: 18px;
        padding: 1.25rem 1.35rem; box-shadow: 0 8px 24px rgba(29, 48, 83, .045);
        margin-bottom: 1rem;
    }
    .section-heading { font-family: 'Manrope', sans-serif; font-weight: 800; font-size: 1.15rem; color: #172744; margin-bottom: .25rem; }
    .section-sub { color: #64748b; font-size: .9rem; margin-bottom: .9rem; }
    .metric-card { background: white; border: 1px solid #e3eaf5; border-radius: 16px; padding: 1rem 1.1rem; }
    .metric-label { font-size: .8rem; color: #64748b; font-weight: 600; }
    .metric-value { font-family: 'Manrope', sans-serif; font-size: 1.6rem; font-weight: 800; color: #172744; margin-top: .25rem; }
    .result-pass { background: #eafaf1; border: 1px solid #b9edce; border-radius: 16px; padding: 1.1rem 1.3rem; }
    .result-fail { background: #fff0f0; border: 1px solid #f7caca; border-radius: 16px; padding: 1.1rem 1.3rem; }
    .result-title { font-family: 'Manrope', sans-serif; font-size: 1.35rem; font-weight: 800; }
    .muted-note { color: #64748b; font-size: .82rem; }
    div.stButton > button[kind="primary"] {
        border-radius: 12px; min-height: 3rem; font-weight: 700;
        background: linear-gradient(100deg, #2456a6, #3677dc); border: 0;
    }
    [data-testid="stSidebar"] { background: #f0f4fb; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_bundle():
    """Load the pre-trained estimator and its encoders once per app session."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {MODEL_PATH.name}. Keep model.pkl in the same folder as app.py."
        )
    bundle = joblib.load(MODEL_PATH)
    required = {"model", "encoders", "columns"}
    if not required.issubset(bundle):
        raise ValueError("The model file is missing required components.")
    return bundle


with st.sidebar:
    st.markdown("## 🎓 Project overview")
    st.write("An interactive machine-learning demo that estimates a student's likely exam outcome.")
    st.markdown("---")
    st.markdown("**Inputs used**")
    st.caption("Study hours · Attendance · Previous score · Extracurricular activities · Home internet access")
    st.markdown("---")
    st.markdown("**Built by**")
    st.write("Ajim Sikandar Mujawar")
    st.caption("Data Science & AI | Machine Learning")
    st.markdown("---")
    st.caption("This is an educational prediction, not a guaranteed result or a substitute for teacher guidance.")

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Machine Learning • Interactive Demo</div>
      <h1>Student Performance Predictor</h1>
      <p>Explore how study habits and learning resources relate to a predicted exam outcome. Enter a student's details to generate an instant model estimate.</p>
      <div class="hero-byline">Designed and developed by <strong>Ajim Sikandar Mujawar</strong></div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    bundle = load_model_bundle()
    model = bundle["model"]
    encoders = bundle["encoders"]
    columns = list(bundle["columns"])
except Exception as exc:
    st.error("The prediction model could not be loaded.")
    st.exception(exc)
    st.stop()

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-heading">🧑‍🎓 Student profile</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Enter the details below, then select Predict outcome.</div>', unsafe_allow_html=True)

numeric_config = {
    "study_hours": {
        "label": "Study hours per week",
        "min": 0.0,
        "max": 100.0,
        "default": 10.0,
        "step": 1.0,
        "help": "Approximate total hours spent studying in a typical week.",
    },
    "attendance_pct": {
        "label": "Attendance",
        "min": 0.0,
        "max": 100.0,
        "default": 75.0,
        "step": 1.0,
        "help": "Percentage of classes attended.",
    },
    "previous_score": {
        "label": "Previous exam score",
        "min": 0.0,
        "max": 100.0,
        "default": 60.0,
        "step": 1.0,
        "help": "Previous score on a 0–100 scale.",
    },
    "extra_curricular": {"label": "Extracurricular activities"},
    "internet_access": {"label": "Internet access at home"},
}

values = {}
left, right = st.columns(2, gap="large")
for idx, feature in enumerate(columns):
    target = left if idx % 2 == 0 else right
    config = numeric_config.get(feature, {})
    label = config.get("label", feature.replace("_", " ").title())
    with target:
        if feature in encoders:
            options = list(encoders[feature].classes_)
            values[feature] = st.selectbox(
                label,
                options=options,
                index=options.index("Yes") if "Yes" in options else 0,
                help="Choose the option that best describes the student.",
                key=f"field_{feature}",
            )
        else:
            values[feature] = st.number_input(
                label,
                min_value=config.get("min", 0.0),
                max_value=config.get("max", 100.0),
                value=config.get("default", 0.0),
                step=config.get("step", 1.0),
                help=config.get("help", ""),
                key=f"field_{feature}",
            )
st.markdown('</div>', unsafe_allow_html=True)

predict_col, reset_col = st.columns([3, 1])
with predict_col:
    predict_clicked = st.button("✨ Predict exam outcome", type="primary", use_container_width=True)
with reset_col:
    reset_clicked = st.button("Reset form", use_container_width=True)

if reset_clicked:
    # Streamlit widget state is reset by rerunning with defaults on the next interaction.
    for feature in columns:
        st.session_state.pop(f"field_{feature}", None)
    st.rerun()

if predict_clicked:
    try:
        encoded_row = {}
        for feature in columns:
            if feature in encoders:
                encoded_row[feature] = encoders[feature].transform([values[feature]])[0]
            else:
                encoded_row[feature] = values[feature]

        input_df = pd.DataFrame([encoded_row], columns=columns)
        prediction = str(model.predict(input_df)[0])
        class_probabilities = model.predict_proba(input_df)[0]
        class_names = [str(name) for name in model.classes_]
        probability_map = {
            class_name: float(probability)
            for class_name, probability in zip(class_names, class_probabilities)
        }
        predicted_probability = probability_map.get(prediction, max(probability_map.values()))
        confidence = predicted_probability * 100

        st.markdown("---")
        st.markdown("## Prediction result")
        result_class = "result-pass" if prediction.strip().lower() == "pass" else "result-fail"
        result_icon = "✅" if prediction.strip().lower() == "pass" else "📘"
        st.markdown(
            f'<div class="{result_class}"><div class="result-title">{result_icon} Predicted outcome: {prediction}</div>'
            f'<div style="margin-top:.3rem">Model confidence for this prediction: <strong>{confidence:.1f}%</strong></div></div>',
            unsafe_allow_html=True,
        )
        st.progress(max(0.0, min(1.0, predicted_probability)))
        st.write("### Class probability breakdown")
        prob_cols = st.columns(len(probability_map))
        for (class_name, probability), target in zip(probability_map.items(), prob_cols):
            target.markdown(
                f'<div class="metric-card"><div class="metric-label">{class_name} probability</div>'
                f'<div class="metric-value">{probability * 100:.1f}%</div></div>',
                unsafe_allow_html=True,
            )
        with st.expander("Review the submitted details"):
            display_values = {
                numeric_config.get(feature, {}).get("label", feature.replace("_", " ").title()): values[feature]
                for feature in columns
            }
            st.dataframe(pd.DataFrame([display_values]), use_container_width=True, hide_index=True)
        st.caption("Probability values are model estimates and may not be calibrated. They should not be interpreted as certainty.")
    except Exception as exc:
        st.error("A prediction could not be generated. Please check the inputs and model dependencies.")
        st.exception(exc)

st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; padding:.6rem 0 0 0; color:#64748b; font-size:.83rem;">
      Built with Streamlit, Pandas and Scikit-learn<br>
      <strong>Ajim Sikandar Mujawar</strong> · Student Result Predictor · For educational demonstration only
    </div>
    """,
    unsafe_allow_html=True,
)

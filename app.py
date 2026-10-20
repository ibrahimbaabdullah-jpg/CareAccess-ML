"""CareAccess ML: a privacy-aware public interface to saved trained models."""

import streamlit as st

from careaccess.inference import load_bundle
from careaccess.research import render_research
from careaccess.session import prepare_batch_authorization
from careaccess.ui.about import render_about
from careaccess.ui.home import render_home
from careaccess.ui.models import render_comparison, render_evaluation
from careaccess.ui.outreach import render_planner
from careaccess.ui.prediction import render_batch, render_live

RENDERERS = {
    "Home / Dashboard": render_home,
    "Live Prediction": render_live,
    "Batch Prediction": render_batch,
    "Outreach Planner": render_planner,
    "Model Comparison": render_comparison,
    "Model Evaluation": render_evaluation,
    "Research / Vitória": lambda bundle: render_research(),
    "About / Documentation": render_about,
}
PAGES = list(RENDERERS)

st.set_page_config(page_title="CareAccess ML", page_icon="📅", layout="wide")
st.markdown("""<style>
.block-container {max-width: 1200px; padding-top: 2rem;}
[data-testid="stMetric"] {background:#edf4f6; padding:1rem; border-radius:12px;}
h1,h2,h3 {letter-spacing:-.025em;}
@media(max-width:640px){.block-container{padding-left:1rem;padding-right:1rem;}}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def trained_bundle():
    return load_bundle()


st.sidebar.title("CareAccess ML")
st.sidebar.caption("Appointment outreach planning")
page = st.sidebar.radio("Explore", PAGES)
st.sidebar.info("Research display: real trained models and aggregate findings, without underlying patient records. Submitted inputs must be permitted, nonsensitive and non-identifying. See the input-page notices before entering data.")
st.sidebar.caption("Supportive reminders only. Never cancel appointments, restrict care, or penalize patients using a score.")
st.title("CareAccess ML")
st.caption("Missed-Appointment Prediction & Outreach Planner")

try:
    bundle = trained_bundle()
except Exception:
    st.error("The verified model bundle could not be loaded. Predictions are unavailable. Please ask the maintainer to check the model files and runtime logs.")
    st.stop()
report = bundle["report"]
default_test_result = next((row for row in report.get("metrics", []) if row.get("model") == bundle["default_model"]), {})
if page in {"Batch Prediction", "Outreach Planner"}:
    prepare_batch_authorization(st.session_state)
if default_test_result.get("roc_auc", 1.0) < 0.60:
    st.warning(f"Important result: the original deployed default model's final-test ROC-AUC is {default_test_result['roc_auc']:.3f}, close to chance (0.500). This release demonstrates working trained-model inference, but does not establish effective outreach targeting. Do not use its rankings for real patient decisions.")

RENDERERS[page](bundle)

st.divider()
st.caption("CareAccess ML · Historical-data evaluation and saved-model inference · Human review required")

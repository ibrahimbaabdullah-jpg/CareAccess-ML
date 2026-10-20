"""Shared widgets, labels and notices for the Streamlit pages."""

import pandas as pd
import streamlit as st

from careaccess.inference import model_names, predict, scoring_model_names
from careaccess.quality import input_quality_warnings
from careaccess.validation import validate_frame

RESEARCH_DEMO_URL = "https://ibrahimbaabdullah-jpg.github.io/CareAccess-ML/"
HOST_INPUT_NOTICE = ("For inputs submitted to this public host: exclude identifiers and sensitive personal information, including patient-level health information. "
                     "Do not upload patient records. Public availability, removing names, or checking a consent box "
                     "does not establish hosting suitability. Source-data evaluation stays offline. "
                     "Real-patient use requires a separately governed deployment and review.")
LABELS = {"logistic": "Logistic Regression", "logistic_regression": "Logistic Regression", "gradient_boosting": "Gradient Boosting", "hist_gradient_boosting": "Gradient Boosting", "pytorch": "PyTorch Neural Network", "neural_network": "PyTorch Neural Network", "baseline": "Constant baseline"}

def label(name):
    return LABELS.get(name, str(name) if " " in str(name) else str(name).replace("_", " ").title())


def model_picker(bundle, key, scoring=True):
    names = scoring_model_names(bundle) if scoring else list(model_names(bundle))
    default = bundle["default_model"]
    return st.selectbox("Trained model", names, index=names.index(default) if default in names else 0, format_func=label, key=key)


def empty_template():
    return pd.DataFrame(columns=["appointment_id", "appointment_datetime", "specialty", "age"])


def validate_and_score(frame, model, bundle):
    clean, errors = validate_frame(frame)
    if not errors.empty:
        st.error("Please correct these problems before scoring. No appointments have been scored.")
        st.dataframe(errors, hide_index=True, width="stretch")
        return None
    probabilities = predict(clean, model_name=model, bundle=bundle)
    result = clean.copy()
    result["miss_probability"] = probabilities
    result["model"] = model
    result["input_quality_warnings"] = input_quality_warnings(clean, bundle["feature_contract"])
    return result


def show_scores(frame):
    st.dataframe(frame, hide_index=True, width="stretch", column_config={"miss_probability": st.column_config.NumberColumn("Estimated non-attendance probability", format="%.4f")})
    st.caption("Probabilities are between 0 and 1: 0.30 means an estimated 30%. These are uncertain estimates from historical data, not a statement about a person's intentions.")


def metric_rows(report):
    metrics = report.get("metrics", {})
    if isinstance(metrics, list):
        return pd.DataFrame(metrics)
    return pd.DataFrame([{"model": name, **values} for name, values in metrics.items()])


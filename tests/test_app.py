"""Real-model UI tests. Positive records are private source-derived fixtures.

Set CAREACCESS_TEST_DATA to a locally reviewed, non-published input CSV.
No fabricated patient cohort is generated or bundled. Source-dependent parity
checks skip explicitly if this private fixture is unavailable.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from careaccess.inference import load_bundle, model_names, predict, scoring_model_names

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def real_inputs():
    path = os.environ.get("CAREACCESS_TEST_DATA")
    if not path or not Path(path).is_file():
        pytest.skip(
            "Private source-derived fixture required for real-input UI parity; no synthetic records substituted"
        )
    return pd.read_csv(path, dtype={"appointment_id": str})


def app():
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()


def select_page(at, name):
    at.sidebar.radio[0].set_value(name).run()
    assert not at.exception
    return at


def test_all_pages_render_with_real_models():
    at = app()
    assert not at.exception
    assert not at.error
    for page in [
        "Live Prediction",
        "Batch Prediction",
        "Outreach Planner",
        "Model Comparison",
        "Model Evaluation",
        "Research / Vitória",
        "About / Documentation",
    ]:
        select_page(at, page)
        assert not at.error


def test_single_form_is_blank_and_requires_privacy_confirmation():
    at = select_page(app(), "Live Prediction")
    assert at.date_input[0].value is None
    assert at.time_input[0].value is None
    assert at.number_input[0].value is None
    assert at.selectbox[1].value is None
    at.button[0].click().run()
    assert not at.exception
    assert any("confirm" in item.value.lower() for item in at.warning)
    assert not at.metric


def test_missing_appointment_time_rejected():
    at = select_page(app(), "Live Prediction")
    at.checkbox(key="live_confirm").check()
    at.button[0].click().run()
    assert not at.exception
    assert any("date and time" in item.value for item in at.error)


@pytest.mark.parametrize("model", ["Logistic Regression", "PyTorch Neural Network"])
def test_live_inference_matches_offline_inference(model, real_inputs):
    bundle = load_bundle()
    assert model in scoring_model_names(bundle)
    row = real_inputs.dropna().iloc[0]
    moment = pd.Timestamp(row.appointment_datetime)
    at = select_page(app(), "Live Prediction")
    at.selectbox(key="live_model").set_value(model).run()
    at.date_input[0].set_value(moment.date())
    at.time_input[0].set_value(moment.time())
    at.number_input[0].set_value(float(row.age))
    at.selectbox[1].set_value(row.specialty)
    at.checkbox(key="live_confirm").check()
    at.button[0].click().run()
    assert not at.exception
    assert not at.error
    displayed = at.session_state["last_single_prediction"]
    frame = pd.DataFrame(
        [{key: displayed[key] for key in ["appointment_id", "appointment_datetime", "age", "specialty"]}]
    )
    offline = predict(frame, model_name=model, bundle=bundle)[0]
    assert np.isclose(displayed["miss_probability"], offline, rtol=1e-7, atol=1e-8)
    assert at.metric[0].value == f"{offline:.1%}"


def test_real_scored_schedule_planner_zero_and_limited_capacity(real_inputs):
    scores = real_inputs.copy()
    bundle = load_bundle()
    scores["miss_probability"] = predict(scores, bundle=bundle)
    scores["model"] = bundle["default_model"]
    at = app()
    at.session_state["batch_scores"] = scores
    at.session_state["batch_authorized"] = True
    select_page(at, "Outreach Planner")
    at.number_input[0].set_value(0).run()
    assert not at.exception
    assert at.metric[0].value == "0"
    at.number_input[0].set_value(min(3, len(scores))).run()
    assert not at.exception
    assert at.metric[0].value == str(min(3, len(scores)))


def test_batch_has_no_generated_schedule_and_gates_upload():
    at = select_page(app(), "Batch Prediction")
    assert "batch_scores" not in at.session_state
    assert len(at.get("file_uploader")) == 0
    assert not at.dataframe
    at.checkbox(key="batch_confirm").check().run()
    assert len(at.get("file_uploader")) == 1
    assert not at.exception


def test_weak_performance_is_disclosed_on_every_page():
    at = app()
    for page in [
        "Home / Dashboard",
        "Live Prediction",
        "Batch Prediction",
        "Outreach Planner",
        "Model Comparison",
        "Model Evaluation",
        "Research / Vitória",
        "About / Documentation",
    ]:
        select_page(at, page)
        assert any("close to chance" in item.value for item in at.warning)


def test_negative_calibration_slope_is_disclosed():
    at = select_page(app(), "Model Evaluation")
    at.selectbox(key="evaluation_model").set_value("Gradient Boosting").run()
    assert any("negative" in item.value and "reverses" in item.value for item in at.warning)


def test_reversed_calibration_model_is_evaluation_only():
    bundle = load_bundle()
    assert "Gradient Boosting" in model_names(bundle)
    assert "Gradient Boosting" not in scoring_model_names(bundle)
    at = select_page(app(), "Live Prediction")
    assert "Gradient Boosting" not in at.selectbox(key="live_model").options
    at = select_page(at, "Model Evaluation")
    assert "Gradient Boosting" in at.selectbox(key="evaluation_model").options

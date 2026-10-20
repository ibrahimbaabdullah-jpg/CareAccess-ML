"""Inference and app plumbing on generated rows, so these paths run in public CI.

The private actual-source suites remain the evidence for real inputs; see
tests/conftest.py for what generated rows can and cannot show.
"""

import io
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from careaccess.csv_input import read_appointments_csv
from careaccess.inference import load_bundle, model_names, predict, scoring_model_names
from careaccess.planner import plan_outreach
from careaccess.validation import INPUT_COLUMNS, make_features, validate_frame

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bundle():
    return load_bundle()


def test_generated_rows_pass_validation(generated_schedule):
    clean, errors = validate_frame(generated_schedule)
    assert errors.empty
    assert len(clean) == len(generated_schedule)


@pytest.mark.parametrize(
    "name", ["Constant baseline", "Logistic Regression", "Gradient Boosting", "PyTorch Neural Network"]
)
def test_every_model_scores_batch_single_and_reload_consistently(generated_schedule, bundle, name):
    batch = predict(generated_schedule, name, bundle)
    single = np.concatenate(
        [predict(generated_schedule.iloc[[i]], name, bundle) for i in range(len(generated_schedule))]
    )
    reloaded = predict(generated_schedule, name, load_bundle())
    assert batch.shape == (len(generated_schedule),)
    assert np.all((batch >= 0) & (batch <= 1))
    np.testing.assert_allclose(batch, single, rtol=0, atol=1e-6)
    np.testing.assert_allclose(batch, reloaded, rtol=0, atol=1e-6)
    if name != "Constant baseline":
        assert np.ptp(batch) > 1e-5


def test_csv_round_trip_scores_match_table(generated_schedule, bundle):
    parsed = read_appointments_csv(generated_schedule.to_csv(index=False).encode())
    for name in model_names(bundle):
        np.testing.assert_allclose(
            predict(parsed, name, bundle), predict(generated_schedule, name, bundle), rtol=0, atol=1e-12
        )


def test_reversed_calibration_inverts_raw_ranking(generated_schedule, bundle):
    x = bundle["preprocessor"].transform(make_features(validate_frame(generated_schedule)[0])).astype("float32")
    raw = bundle["models"]["Gradient Boosting"].predict_proba(x)[:, 1]
    calibrated = predict(generated_schedule, "Gradient Boosting", bundle)
    order = np.argsort(raw)
    assert bundle["manifest"]["calibrators"]["Gradient Boosting"]["coef"] < 0
    assert np.all(np.diff(calibrated[order]) <= 1e-12)


class Upload(io.BytesIO):
    def __init__(self, payload):
        super().__init__(payload)
        self.size = len(payload)
        self.name = "generated.csv"


def run(at, payload=None):
    downloads = {}
    original = st.download_button

    def capture(label, data, *args, **kwargs):
        downloads[label] = data
        return original(label, data, *args, **kwargs)

    uploader = patch(
        "streamlit.file_uploader", side_effect=lambda *a, **k: Upload(payload) if payload is not None else None
    )
    with uploader, patch("streamlit.download_button", side_effect=capture):
        at.run()
    assert not at.exception
    return downloads


def open_page(name):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    at.sidebar.radio[0].set_value(name).run()
    assert not at.exception
    return at


@pytest.mark.parametrize("model", ["Constant baseline", "Logistic Regression", "PyTorch Neural Network"])
def test_live_prediction_matches_offline(generated_schedule, bundle, model):
    row = generated_schedule.dropna().iloc[0]
    moment = pd.Timestamp(row.appointment_datetime)
    at = open_page("Live Prediction")
    at.selectbox(key="live_model").set_value(model)
    at.date_input[0].set_value(moment.date())
    at.time_input[0].set_value(moment.time())
    at.number_input[0].set_value(float(row.age))
    at.selectbox[1].set_value(row.specialty)
    at.checkbox(key="live_confirm").check()
    at.button[0].click()
    downloads = run(at)
    assert not at.error
    shown = at.session_state["last_single_prediction"]
    frame = pd.DataFrame([{key: shown[key] for key in INPUT_COLUMNS}])
    offline = predict(frame, model, bundle)[0]
    assert np.isclose(shown["miss_probability"], offline, rtol=1e-7, atol=1e-8)
    assert at.metric[0].value == f"{offline:.1%}"
    exported = pd.read_csv(io.BytesIO(downloads["Download this prediction"]))
    assert np.isclose(exported.miss_probability[0], offline, rtol=0, atol=1e-12)


@pytest.mark.parametrize("model", ["Constant baseline", "Logistic Regression", "PyTorch Neural Network"])
def test_batch_scores_export_and_planner_match_offline(generated_schedule, bundle, model):
    payload = generated_schedule.to_csv(index=False).encode()
    at = open_page("Batch Prediction")
    at.checkbox(key="batch_confirm").check()
    run(at, payload)
    at.selectbox(key="batch_model").set_value(model)
    next(item for item in at.button if item.label == "Validate and score appointments").click()
    downloads = run(at, payload)
    assert not at.error
    scored = at.session_state["batch_scores"]
    np.testing.assert_allclose(scored.miss_probability, predict(generated_schedule, model, bundle), rtol=0, atol=1e-6)
    exported = pd.read_csv(io.BytesIO(downloads["Download scored schedule"]), dtype={"appointment_id": str})
    np.testing.assert_allclose(exported.miss_probability, scored.miss_probability, rtol=0, atol=1e-12)

    at.sidebar.radio[0].set_value("Outreach Planner")
    run(at)
    at.number_input[0].set_value(7)
    downloads = run(at)
    assert at.metric[0].value == "7"
    review = pd.read_csv(io.BytesIO(downloads["Download outreach review list"]), dtype={"appointment_id": str})
    expected = plan_outreach(scored, 7)
    assert review.appointment_id.tolist() == expected.appointment_id.tolist()
    assert review.priority.tolist() == list(range(1, 8))


def test_live_picker_offers_only_scoring_models(bundle):
    at = open_page("Live Prediction")
    assert at.selectbox(key="live_model").options == scoring_model_names(bundle)

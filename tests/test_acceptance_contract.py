"""Application acceptance contracts using private, actual-source inputs only.

Set CAREACCESS_TEST_DATA to the existing reviewed source-derived fixture.
No patient cohort or positive-input substitute is generated or distributed.
Download checks inspect the actual bytes passed to Streamlit's download widget;
they do not claim browser-side HTTP download verification.
"""

import io
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from careaccess.inference import predict
from careaccess.planner import plan_outreach
from careaccess.validation import INPUT_COLUMNS

ROOT = Path(os.environ.get("CAREACCESS_APP_ROOT", Path(__file__).resolve().parents[1]))
# Gradient Boosting is evaluation-only: its negative calibration slope reverses rankings.
MODELS = ["Constant baseline", "Logistic Regression", "PyTorch Neural Network"]


@pytest.fixture
def actual_inputs():
    path = os.environ.get("CAREACCESS_TEST_DATA")
    if not path or not Path(path).is_file():
        pytest.skip("Private actual-source adult fixture required; no synthetic rows substituted")
    return pd.read_csv(path, dtype={"appointment_id": str})


class Upload(io.BytesIO):
    def __init__(self, payload):
        super().__init__(payload)
        self.size = len(payload)
        self.name = "private_fixture.csv"


def app():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception
    return at


def page(at, name):
    at.sidebar.radio[0].set_value(name).run()
    assert not at.exception
    return at


def run(at, payload=None):
    downloads = {}
    original = st.download_button

    def capture(label, data, *args, **kwargs):
        downloads[label] = data
        return original(label, data, *args, **kwargs)

    # Fresh stream on each rerun matches Streamlit's readable uploaded object.
    with (
        patch("streamlit.file_uploader", side_effect=lambda *a, **k: Upload(payload) if payload is not None else None),
        patch("streamlit.download_button", side_effect=capture),
    ):
        at.run()
    assert not at.exception
    return downloads


def batch(frame, model="Logistic Regression"):
    at = page(app(), "Batch Prediction")
    at.checkbox(key="batch_confirm").check()
    payload = frame.to_csv(index=False).encode()
    run(at, payload)
    at.selectbox(key="batch_model").set_value(model)
    next(item for item in at.button if item.label == "Validate and score appointments").click()
    downloads = run(at, payload)
    assert not at.error
    return at, downloads


def csv_matches(payload, frame):
    assert isinstance(payload, bytes) and payload.startswith(b"\xef\xbb\xbf")
    parsed = pd.read_csv(io.BytesIO(payload), dtype={"appointment_id": str})
    pd.testing.assert_frame_equal(
        parsed, frame.reset_index(drop=True), check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12
    )
    return parsed


def edit_first_review(at, value="Ready for supportive outreach"):
    key = at.dataframe[0].key or at.dataframe[0].proto.id
    at.session_state[key] = {"edited_rows": {"0": {"review_status": value}}, "added_rows": [], "deleted_rows": []}
    return run(at)


@pytest.mark.parametrize("model", MODELS)
def test_batch_export_bytes_match_display_and_offline(actual_inputs, model):
    at, downloads = batch(actual_inputs, model)
    displayed = at.dataframe[-1].value
    pd.testing.assert_frame_equal(displayed, at.session_state["batch_scores"])
    np.testing.assert_allclose(displayed.miss_probability, predict(actual_inputs, model), rtol=0, atol=1e-6)
    csv_matches(downloads["Download scored schedule"], displayed)


@pytest.mark.parametrize("model", MODELS)
def test_single_export_bytes_match_display_and_offline(actual_inputs, model):
    row = actual_inputs.dropna().iloc[0]
    moment = pd.Timestamp(row.appointment_datetime)
    at = page(app(), "Live Prediction")
    at.selectbox(key="live_model").set_value(model).run()
    at.date_input[0].set_value(moment.date())
    at.time_input[0].set_value(moment.time())
    at.number_input[0].set_value(float(row.age))
    at.selectbox[1].set_value(row.specialty)
    at.checkbox(key="live_confirm").check()
    at.button[0].click()
    downloads = run(at)
    assert not at.error
    displayed = at.dataframe[0].value
    np.testing.assert_allclose(displayed.miss_probability, predict(displayed[INPUT_COLUMNS], model), rtol=0, atol=1e-6)
    csv_matches(downloads["Download this prediction"], displayed)


@pytest.mark.parametrize("capacity", [0, 5, 24])
def test_planner_export_bytes_match_capacity_display(actual_inputs, capacity):
    at, _ = batch(actual_inputs)
    page(at, "Outreach Planner")
    capacity = min(capacity, len(actual_inputs))
    at.number_input[0].set_value(capacity)
    downloads = run(at)
    expected = plan_outreach(at.session_state["batch_scores"], capacity)
    if capacity:
        expected["review_status"] = "Not reviewed"
        pd.testing.assert_frame_equal(at.dataframe[0].value, expected)
    csv_matches(downloads["Download outreach review list"], expected)
    assert at.metric[0].value == str(capacity)


@pytest.mark.parametrize("action", ["immediate", "navigate", "capacity", "zero_then_restore"])
def test_review_status_survives_and_is_exported(actual_inputs, action):
    at, _ = batch(actual_inputs)
    page(at, "Outreach Planner")
    at.number_input[0].set_value(5)
    run(at)
    before = plan_outreach(at.session_state["batch_scores"], 5)
    target = before.appointment_id.iloc[0]
    downloads = edit_first_review(at)
    if action == "navigate":
        page(at, "Model Comparison")
        page(at, "Outreach Planner")
        downloads = run(at)
    elif action == "capacity":
        at.number_input[0].set_value(6)
        downloads = run(at)
    elif action == "zero_then_restore":
        at.number_input[0].set_value(0)
        run(at)
        at.number_input[0].set_value(5)
        downloads = run(at)
    parsed = pd.read_csv(io.BytesIO(downloads["Download outreach review list"]))
    assert parsed.loc[parsed.appointment_id.eq(target), "review_status"].iloc[0] == "Ready for supportive outreach"
    expected = plan_outreach(at.session_state["batch_scores"], len(parsed))
    pd.testing.assert_series_equal(parsed.appointment_id, expected.appointment_id)
    np.testing.assert_allclose(parsed.miss_probability, expected.miss_probability, rtol=0, atol=1e-12)


@pytest.mark.parametrize("action", ["clear", "revoke"])
def test_clear_or_revoke_removes_upload_scores_and_reviews(actual_inputs, action):
    at, _ = batch(actual_inputs)
    page(at, "Outreach Planner")
    edit_first_review(at)
    page(at, "Batch Prediction")
    at.checkbox(key="batch_confirm").check()
    run(at, actual_inputs.to_csv(index=False).encode())
    if action == "clear":
        next(item for item in at.button if item.label == "Clear schedule and results").click()
    else:
        at.checkbox(key="batch_confirm").uncheck()
    downloads = run(at, actual_inputs.to_csv(index=False).encode())
    assert "batch_scores" not in at.session_state
    assert "batch_review_statuses" not in at.session_state
    assert not at.checkbox(key="batch_confirm").value
    assert not at.get("file_uploader") and not at.dataframe
    assert "Download scored schedule" not in downloads
    page(at, "Outreach Planner")
    assert not at.metric and not at.dataframe


def test_failed_new_scoring_clears_prior_results_and_reviews(actual_inputs):
    at, _ = batch(actual_inputs)
    page(at, "Outreach Planner")
    edit_first_review(at)
    page(at, "Batch Prediction")
    at.checkbox(key="batch_confirm").check()
    corrupted = actual_inputs.copy()
    corrupted.loc[0, "specialty"] = "UNSUPPORTED"
    payload = corrupted.to_csv(index=False).encode()
    run(at, payload)
    next(item for item in at.button if item.label == "Validate and score appointments").click()
    run(at, payload)
    assert at.error and "batch_scores" not in at.session_state
    assert "batch_review_statuses" not in at.session_state
    page(at, "Outreach Planner")
    assert not at.metric


@pytest.mark.parametrize("kind", ["empty", "encoding", "quotes", "missing_column", "extra_column"])
def test_malformed_csv_rejected_before_preview_or_score(kind):
    payloads = {
        "empty": b"",
        "encoding": b"\xff\xfe\xff",
        "quotes": b'"appointment_id,appointment_datetime,specialty,age\n',
        "missing_column": b"appointment_id,appointment_datetime,specialty\n",
        "extra_column": b"appointment_id,appointment_datetime,specialty,age,patient_name\n",
    }
    at = page(app(), "Batch Prediction")
    at.checkbox(key="batch_confirm").check()
    run(at, payloads[kind])
    assert at.error and not at.dataframe
    assert not any(item.label == "Validate and score appointments" for item in at.button)
    assert "batch_scores" not in at.session_state


def test_header_only_file_produces_no_invented_rows():
    at, downloads = batch(pd.DataFrame(columns=INPUT_COLUMNS))
    assert at.session_state["batch_scores"].empty
    csv_matches(downloads["Download scored schedule"], at.dataframe[-1].value)
    page(at, "Outreach Planner")
    assert at.metric[0].value == "0"


def test_missing_field_warnings_survive_export_and_planning(actual_inputs):
    at, downloads = batch(actual_inputs)
    scored = at.session_state["batch_scores"]
    missing = actual_inputs.specialty.isna() | actual_inputs.specialty.fillna("").str.strip().eq("")
    assert missing.any(), "Source fixture needs at least one genuine missing specialty for this check"
    assert scored.loc[missing, "input_quality_warnings"].str.contains("Specialty missing").all()
    csv_matches(downloads["Download scored schedule"], scored)
    page(at, "Outreach Planner")
    assert "input_quality_warnings" in at.dataframe[0].value


def test_model_load_failure_is_explicit_and_non_scoring():
    st.cache_resource.clear()
    try:
        with patch("careaccess.inference.load_bundle", side_effect=ValueError("Isolated load-failure test")):
            at = app()
        assert any("could not be loaded" in error.value for error in at.error)
        assert not at.metric and not at.dataframe
    finally:
        st.cache_resource.clear()


def test_multiday_batch_capacity_is_total_and_ties_are_deterministic(actual_inputs):
    assert pd.to_datetime(actual_inputs.appointment_datetime).dt.date.nunique() > 1
    at, _ = batch(actual_inputs, model="Constant baseline")
    page(at, "Outreach Planner")
    at.number_input[0].set_value(5)
    downloads = run(at)
    parsed = pd.read_csv(io.BytesIO(downloads["Download outreach review list"]))
    assert len(parsed) == 5
    assert parsed.appointment_id.tolist() == sorted(actual_inputs.appointment_id.tolist())[:5]
    assert any("entire uploaded batch" in item.value for item in at.caption)


def test_daily_research_policy_is_distinguished_from_interactive_planner():
    at = page(app(), "Model Evaluation")
    assert any("different selection policies" in item.value for item in at.caption)


def test_planner_capacity_and_review_survive_home_navigation(actual_inputs):
    at, _ = batch(actual_inputs)
    page(at, "Outreach Planner")
    at.number_input[0].set_value(5)
    run(at)
    edit_first_review(at)
    page(at, "Home / Dashboard")
    page(at, "Outreach Planner")
    downloads = run(at)
    assert at.number_input[0].value == 5
    assert at.metric[0].value == "5"
    parsed = pd.read_csv(io.BytesIO(downloads["Download outreach review list"]))
    assert len(parsed) == 5
    assert parsed.review_status.iloc[0] == "Ready for supportive outreach"


def test_empty_batch_authorization_widget_survives_navigation_until_revoked():
    at = page(app(), "Batch Prediction")
    assert not at.checkbox(key="batch_confirm").value
    at.checkbox(key="batch_confirm").check().run()
    page(at, "Home / Dashboard")
    page(at, "Batch Prediction")
    assert at.checkbox(key="batch_confirm").value is True
    at.checkbox(key="batch_confirm").uncheck().run()
    page(at, "Home / Dashboard")
    page(at, "Batch Prediction")
    assert at.checkbox(key="batch_confirm").value is False


@pytest.mark.parametrize("destination", ["Batch Prediction", "Outreach Planner"])
def test_legacy_unconsented_results_not_exposed_on_either_route(actual_inputs, destination):
    at = app()
    legacy = actual_inputs.copy()
    legacy["miss_probability"] = predict(actual_inputs)
    at.session_state["batch_scores"] = legacy
    # An old session without durable consent cannot infer it from retained data.
    page(at, destination)
    downloads = run(at)
    assert "batch_scores" not in at.session_state
    assert not at.dataframe and not at.metric
    assert "Download scored schedule" not in downloads
    assert "Download outreach review list" not in downloads


def test_authorization_survives_navigation_and_explicit_revoke_still_clears(actual_inputs):
    at, _ = batch(actual_inputs)
    page(at, "Outreach Planner")
    at.number_input[0].set_value(5).run()
    page(at, "Home / Dashboard")
    page(at, "Batch Prediction")
    downloads = run(at)
    assert at.checkbox(key="batch_confirm").value is True
    assert at.session_state["batch_authorized"] is True
    assert "Download scored schedule" in downloads
    at.checkbox(key="batch_confirm").uncheck()
    downloads = run(at)
    assert not at.checkbox(key="batch_confirm").value
    assert "batch_scores" not in at.session_state
    assert "batch_capacity" not in at.session_state
    assert "Download scored schedule" not in downloads
    page(at, "Outreach Planner")
    assert not at.number_input and not at.metric


def test_clear_resets_capacity_before_new_schedule_and_shrink_is_safe(actual_inputs):
    at, _ = batch(actual_inputs)
    page(at, "Outreach Planner")
    at.number_input[0].set_value(5).run()
    page(at, "Batch Prediction")
    next(item for item in at.button if item.label == "Clear schedule and results").click()
    run(at)
    assert not at.checkbox(key="batch_confirm").value
    assert "batch_capacity" not in at.session_state
    assert "batch_capacity_input" not in at.session_state
    # A new authorized real-source schedule gets a fresh, bounded default.
    at.checkbox(key="batch_confirm").check()
    run(at, actual_inputs.to_csv(index=False).encode())
    next(item for item in at.button if item.label == "Validate and score appointments").click()
    run(at, actual_inputs.to_csv(index=False).encode())
    page(at, "Outreach Planner")
    assert at.number_input[0].value == min(20, len(actual_inputs))
    # Replace the batch with fewer real rows, without clearing the session.
    page(at, "Batch Prediction")
    payload = actual_inputs.iloc[:3].to_csv(index=False).encode()
    run(at, payload)
    next(item for item in at.button if item.label == "Validate and score appointments").click()
    run(at, payload)
    page(at, "Outreach Planner")
    assert at.number_input[0].value == 3
    assert at.metric[0].value == "3"

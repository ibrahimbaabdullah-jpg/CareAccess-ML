"""Fixture selection on generated prepared rows, so the private fixture's contract is checked in public CI."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from careaccess.fixture import FIXTURE_ROWS, MIN_AGE, select_fixture_rows
from careaccess.validation import INPUT_COLUMNS, SPECIALTIES, validate_frame

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / "models" / "manifest.json").read_text())["feature_contract"]


def make_prepared(rows=600, seed=5):
    """Rows shaped like `prepare_source` output, spanning the training window and the later splits."""
    rng = np.random.default_rng(seed)
    start, end = pd.Timestamp("2016-06-20").value // 10**9, pd.Timestamp("2022-02-25").value // 10**9
    days = pd.to_datetime(rng.integers(start, end, rows), unit="s").normalize()
    stamps = days + pd.to_timedelta(rng.integers(7, 18, rows), unit="h")
    age = rng.integers(2, 95, rows).astype(float)
    specialty = rng.choice(SPECIALTIES, rows).astype(object)
    specialty[rng.random(rows) < 0.1] = np.nan
    return pd.DataFrame(
        {
            "appointment_datetime": stamps.strftime("%Y-%m-%dT%H:%M"),
            "specialty": specialty,
            "age": age,
            "target": rng.integers(0, 2, rows),
            "gender_audit": rng.choice(["M", "F"], rows),
        }
    )


@pytest.fixture
def fixture():
    return select_fixture_rows(make_prepared(), CONTRACT)


def test_row_count_and_columns(fixture):
    assert list(fixture.columns) == INPUT_COLUMNS
    assert len(fixture) == FIXTURE_ROWS


def test_every_row_passes_input_validation(fixture):
    clean, errors = validate_frame(fixture)
    assert errors.empty
    assert len(clean) == FIXTURE_ROWS
    assert fixture.appointment_id.is_unique


def test_requirements_of_the_source_dependent_tests(fixture):
    assert pd.to_datetime(fixture.appointment_datetime).dt.date.nunique() > 1
    assert fixture.specialty.isna().any(), "at least one genuine missing specialty is required"
    assert not fixture.dropna().empty, "at least one row with no missing field is required"


def test_no_child_or_unknown_age_rows(fixture):
    assert fixture.age.notna().all()
    assert (fixture.age >= MIN_AGE).all()


def test_dates_fall_outside_the_training_window(fixture):
    dates = pd.to_datetime(fixture.appointment_datetime)
    inside = dates.between(pd.Timestamp(CONTRACT["train_start"]), pd.Timestamp(CONTRACT["train_end"]))
    assert not inside.any()


def test_no_target_or_demographic_column_survives(fixture):
    assert not {"target", "gender_audit", "no_show", "date_of_birth", "gender"} & set(fixture.columns)


def test_selection_is_deterministic():
    first = select_fixture_rows(make_prepared(), CONTRACT)
    second = select_fixture_rows(make_prepared(), CONTRACT)
    pd.testing.assert_frame_equal(first, second)


def test_too_few_eligible_rows_is_refused():
    prepared = make_prepared()
    prepared["age"] = 4.0
    with pytest.raises(ValueError, match="Too few eligible adult appointments"):
        select_fixture_rows(prepared, CONTRACT)


def test_missing_specialty_requirement_is_refused_when_absent():
    prepared = make_prepared()
    prepared["specialty"] = SPECIALTIES[0]
    with pytest.raises(ValueError, match="Too few eligible adult appointments"):
        select_fixture_rows(prepared, CONTRACT)

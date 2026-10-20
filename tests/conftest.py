"""Generated rows for software-behaviour tests.

These rows exercise parsing, validation, inference plumbing, training mechanics
and the app. They are not patient data, are never written into the repository
and say nothing about model performance: evaluation evidence comes only from
the documented source datasets, and the private actual-source checks still
require CAREACCESS_TEST_DATA.
"""

import numpy as np
import pandas as pd
import pytest

from careaccess.validation import SPECIALTIES


def make_schedule(rows=40, seed=11):
    """Rows in the public CSV template layout, with blanks and several dates."""
    rng = np.random.default_rng(seed)
    days = pd.Timestamp("2021-03-01") + pd.to_timedelta(rng.integers(0, 6, rows), unit="D")
    times = pd.to_timedelta(rng.integers(7, 18, rows), unit="h") + pd.to_timedelta(
        rng.choice([0, 15, 30, 45], rows), unit="min"
    )
    age = rng.integers(1, 95, rows).astype(float)
    age[rng.random(rows) < 0.15] = np.nan
    specialty = rng.choice(SPECIALTIES, rows).astype(object)
    specialty[rng.random(rows) < 0.15] = np.nan
    return pd.DataFrame(
        {
            "appointment_id": [f"G{i:03d}" for i in range(rows)],
            "appointment_datetime": (days + times).strftime("%Y-%m-%dT%H:%M"),
            "specialty": specialty,
            "age": age,
        }
    )


def make_source(rows=4000, seed=7):
    """Rows in the Mendeley source column layout, spanning every chronological split."""
    rng = np.random.default_rng(seed)
    start, end = pd.Timestamp("2016-06-20").value // 10**9, pd.Timestamp("2022-06-30").value // 10**9
    days = pd.to_datetime(rng.integers(start, end, rows), unit="s").normalize()
    hour = rng.integers(7, 18, rows)
    years = rng.integers(1, 90, rows)
    birth = days - pd.to_timedelta(years * 365.25 + rng.integers(0, 300, rows), unit="D")
    specialty = rng.choice(SPECIALTIES, rows).astype(object)
    specialty[rng.random(rows) < 0.06] = np.nan
    risk = 1 / (1 + np.exp(-(-2.0 + 0.05 * (hour - 12) - 0.005 * (years - 40))))
    return pd.DataFrame(
        {
            "appointment_date": days.strftime("%d/%m/%Y"),
            "appointment_time": [f"{h:02d}:30" for h in hour],
            "date_of_birth": birth.strftime("%d/%m/%Y"),
            "specialty": specialty,
            "gender": rng.choice(["M", "F"], rows),
            "no_show": np.where(rng.random(rows) < risk, "yes", "no"),
        }
    )


@pytest.fixture
def generated_schedule():
    return make_schedule()


@pytest.fixture
def generated_source_csv(tmp_path):
    """A temporary source-layout file; prepare_source needs a path to checksum."""
    path = tmp_path / "generated_source.csv"
    make_source().to_csv(path, index=False)
    return path

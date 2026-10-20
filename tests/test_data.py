"""Source preparation on a generated file in the source layout."""

import hashlib

import pandas as pd
import pytest

from careaccess import data


@pytest.fixture
def trusted(monkeypatch):
    def trust(path):
        monkeypatch.setattr(data, "SOURCE_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
        return path

    return trust


def test_unverified_source_is_rejected(generated_source_csv):
    with pytest.raises(ValueError, match="checksum mismatch"):
        data.prepare_source(generated_source_csv)


def test_prepare_source_cleans_and_derives_age(tmp_path, trusted):
    raw = pd.DataFrame(
        {
            "appointment_date": ["02/01/2018", "02/01/2018", "01/01/2018", "03/01/2018", "04/01/2018"],
            "appointment_time": ["10:30", "10:30", "08:00", "09:00", "11:00"],
            "date_of_birth": ["02/01/1988", "02/01/1988", "01/01/2020", "bad", "01/01/2000"],
            "specialty": ["enf", "enf", None, "assist", "assist"],
            "gender": ["F", "F", "M", "M", "F"],
            "no_show": ["yes", "yes", "no", "maybe", "no"],
        }
    )
    path = tmp_path / "source.csv"
    raw.to_csv(path, index=False)
    frame, counts = data.prepare_source(trusted(path))
    assert counts["original_rows"] == 5
    assert counts["exact_duplicate_rows_removed"] == 1
    assert counts["invalid_datetime_or_target_removed"] == 1
    assert counts["clean_rows"] == 3 and counts["missed"] == 1
    assert frame.appointment_datetime.tolist() == ["2018-01-01T08:00", "2018-01-02T10:30", "2018-01-04T11:00"]
    # A birth date after the appointment gives no usable age.
    assert pd.isna(frame.age[0])
    assert frame.age[1] == pytest.approx(30, abs=0.01)
    assert counts["missing_derived_age"] == 1 and counts["missing_specialty"] == 1
    assert "date_of_birth" not in frame and "gender_audit" in frame


def test_generated_source_covers_every_split(generated_source_csv, trusted):
    frame, _ = data.prepare_source(trusted(generated_source_csv))
    parts = data.temporal_split(frame)
    assert sum(len(part) for part in parts.values()) == len(frame)
    assert all(len(part) and part.target.nunique() == 2 for part in parts.values())

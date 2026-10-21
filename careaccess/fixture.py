"""Select actual-source rows for the private test fixture. Row-level records never enter the repository."""

import numpy as np
import pandas as pd

from careaccess.validation import INPUT_COLUMNS, validate_frame

FIXTURE_ROWS = 24
MIN_AGE = 18


def select_fixture_rows(frame: pd.DataFrame, contract: dict, rows: int = FIXTURE_ROWS) -> pd.DataFrame:
    """Adult appointments dated outside the training window, in the public input-template layout.

    The source-dependent tests require exactly `rows` valid rows spanning more than one calendar
    date, at least one genuine missing specialty and at least one row with no missing field. Ages
    below `MIN_AGE` and rows of unknown age are excluded, so no child record can reach the fixture.
    """
    dates = pd.to_datetime(frame.appointment_datetime, format="%Y-%m-%dT%H:%M", errors="raise")
    outside = ~dates.between(pd.Timestamp(contract["train_start"]), pd.Timestamp(contract["train_end"]))
    adult = pd.to_numeric(frame.age, errors="coerce") >= MIN_AGE
    eligible = frame[outside & adult].sort_values("appointment_datetime", kind="stable")
    with_specialty = eligible[eligible.specialty.notna()]
    without_specialty = eligible[eligible.specialty.isna()]
    if len(with_specialty) < rows - 1 or without_specialty.empty:
        raise ValueError(
            f"Too few eligible adult appointments outside the training window: {len(with_specialty)} with a "
            f"specialty and {len(without_specialty)} without, need {rows - 1} and 1."
        )
    spread = np.linspace(0, len(with_specialty) - 1, rows - 1).round().astype(int)
    chosen = pd.concat([with_specialty.iloc[spread], without_specialty.iloc[[len(without_specialty) // 2]]])
    chosen = chosen.sort_values("appointment_datetime", kind="stable").reset_index(drop=True)
    fixture = pd.DataFrame(
        {
            "appointment_id": [f"A{i + 1:03d}" for i in range(len(chosen))],
            "appointment_datetime": chosen.appointment_datetime,
            "specialty": chosen.specialty,
            "age": pd.to_numeric(chosen.age).pipe(np.floor).astype(int),
        }
    )[INPUT_COLUMNS]
    clean, errors = validate_frame(fixture)
    if not errors.empty or len(clean) != rows:
        raise ValueError(f"Selected rows failed input validation: {len(clean)} valid of {rows}.")
    if pd.to_datetime(fixture.appointment_datetime).dt.date.nunique() < 2:
        raise ValueError("The fixture must span more than one calendar date.")
    return fixture

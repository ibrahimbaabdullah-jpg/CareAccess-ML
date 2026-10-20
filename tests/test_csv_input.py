"""Adversarial parser strings only: no generated appointment cohort or scores."""

import csv
import io

import pandas as pd
import pytest

from careaccess.csv_input import MAX_BYTES, MAX_CELL_CHARS, read_appointments_csv
from careaccess.validation import INPUT_COLUMNS

HEADER = ",".join(INPUT_COLUMNS) + "\n"


def test_header_only_template_and_bom():
    for data in (HEADER.encode(), ("\ufeff" + HEADER).encode()):
        frame = read_appointments_csv(data)
        assert frame.empty
        assert list(frame) == INPUT_COLUMNS


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"\xff",
        b"\x00",
        b"\n",
        b"appointment_id,appointment_datetime,specialty,age,patient_name\n",
        b"appointment_id,appointment_datetime,specialty,appointment_id\n",
        b"appointment_id,appointment_datetime,specialty\n",
        (HEADER + ",,,,\n").encode(),
        (HEADER + ",,\n").encode(),
        (HEADER + '"unterminated,,,\n').encode(),
        (HEADER + '"line\nbreak",,,\n').encode(),
        (HEADER + "\x00,,,\n").encode(),
        (HEADER + ("x" * (MAX_CELL_CHARS + 1)) + ",,,\n").encode(),
        (HEADER + ",,,\n" * 5001).encode(),
        b"x" * (MAX_BYTES + 1),
        ("c," * 99999 + "c\n" + "0," * 99999 + "0\n").encode(),
    ],
)
def test_invalid_structure_rejected_before_dataframe_allocation(payload, monkeypatch):
    def allocation_forbidden(*args, **kwargs):
        pytest.fail("A DataFrame was allocated before structural validation completed.")

    monkeypatch.setattr(pd, "DataFrame", allocation_forbidden)
    with pytest.raises(ValueError):
        read_appointments_csv(payload)


def test_bounded_empty_fields_are_preserved_for_semantic_validation():
    # Deliberately invalid empty rows exercise the parser, never model inference.
    frame = read_appointments_csv((HEADER + ",,,\n" * 5000).encode())
    assert frame.shape == (5000, 4)
    assert frame.eq("").all().all()


def test_column_order_and_quoted_field_parsing():
    columns = list(reversed(INPUT_COLUMNS))
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(columns)
    writer.writerow(["", "invalid,quoted", "", "NOT-A-PATIENT"])
    frame = read_appointments_csv(output.getvalue().encode())
    assert list(frame) == columns
    assert frame.iloc[0]["specialty"] == "invalid,quoted"


def test_non_bytes_rejected():
    with pytest.raises(ValueError, match="UTF-8"):
        read_appointments_csv(HEADER)


def test_tokens_are_not_silently_converted_to_numbers_or_missing_values():
    # Missing dates make these invalid appointments; this tests decoding only.
    frame = read_appointments_csv((HEADER + "00001,,,NaN\nNA,,,NULL\n").encode())
    assert frame.appointment_id.tolist() == ["00001", "NA"]
    assert frame.age.tolist() == ["NaN", "NULL"]

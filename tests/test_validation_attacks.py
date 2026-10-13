"""Source-independent malformed inputs; no positive generated patient records."""
import pandas as pd
import pytest

from careaccess.validation import INPUT_COLUMNS, validate_frame


@pytest.mark.parametrize('field,value', [
    ('appointment_datetime', '2026-1-01T01:01'),
    ('appointment_datetime', '2026-01-1T01:01'),
    ('appointment_datetime', '2026-01-01T1:01'),
    ('appointment_datetime', '2026-01-01T01:1'),
    ('appointment_datetime', '2026-01-01 01:01'),
    ('appointment_datetime', '2026-01-01T01:01:00'),
    ('appointment_datetime', '2026-01-01T01:01Z'),
    ('appointment_datetime', '2026-02-30T01:01'),
    ('age', '-1'), ('age', '111'), ('age', 'nan'), ('age', 'inf'),
    ('age', 'not-a-number'), ('specialty', 'unsupported'),
    ('appointment_id', '=1+1'), ('appointment_id', 'contains spaces'),
    ('appointment_id', 'x' * 41),
])
def test_invalid_field_always_reported(field, value):
    invalid = dict.fromkeys(INPUT_COLUMNS, '')
    invalid[field] = value
    clean, errors = validate_frame(pd.DataFrame([invalid]))
    assert clean.empty
    assert field in set(errors.field)


@pytest.mark.parametrize('extra', ['patient_name', 'date_of_birth', 'no_show', 'no_show_reason', 'average_temp_day'])
def test_unsupported_headers_rejected_without_any_source_rows(extra):
    clean, errors = validate_frame(pd.DataFrame(columns=[*INPUT_COLUMNS, extra]))
    assert clean.empty
    assert set(errors.field) == {'columns'}


def test_duplicate_invalid_references_are_reported_without_inference():
    invalid = dict.fromkeys(INPUT_COLUMNS, '')
    clean, errors = validate_frame(pd.DataFrame([invalid, invalid]))
    assert clean.empty
    duplicate_errors = errors[errors.message.eq('Duplicate appointment reference.')]
    assert duplicate_errors.row.tolist() == [1, 2]

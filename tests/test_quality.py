import os
from pathlib import Path

import pandas as pd
import pytest

from careaccess.inference import load_bundle, predict
from careaccess.quality import input_quality_warnings
from careaccess.validation import validate_frame


def test_actual_source_missing_input_flags_do_not_change_predictions():
    path = os.environ.get('CAREACCESS_TEST_DATA')
    if not path or not Path(path).is_file():
        pytest.skip('Private actual-source fixture unavailable; no substitute appointments generated.')
    frame, errors = validate_frame(pd.read_csv(path, dtype={'appointment_id': str}))
    assert errors.empty
    bundle = load_bundle()
    before = predict(frame, bundle=bundle)
    flags = input_quality_warnings(frame, bundle['feature_contract'])
    assert flags.str.contains('Specialty missing').equals(frame.specialty.isna())
    assert flags.str.contains('Age missing').equals(frame.age.isna())
    assert (before == predict(frame, bundle=bundle)).all()
    assert flags.str.contains('outside model training period').all()

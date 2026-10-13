"""Input-quality context accompanying estimates; never changes model inputs."""
import pandas as pd


def input_quality_warnings(frame, contract):
    warnings = pd.Series('', index=frame.index, dtype=str)
    def add(mask, message):
        warnings.loc[mask] = warnings.loc[mask].map(lambda value: value + '; ' + message if value else message)
    add(frame.age.isna(), 'Age missing: training median and missingness indicator used')
    add(frame.specialty.isna(), 'Specialty missing: missing category used')
    dates = pd.to_datetime(frame.appointment_datetime, format='%Y-%m-%dT%H:%M', errors='raise')
    if contract.get('train_start') and contract.get('train_end'):
        outside = ~dates.between(pd.Timestamp(contract['train_start']), pd.Timestamp(contract['train_end']))
        add(outside, 'Appointment date outside model training period; local applicability is unverified')
    return warnings.mask(warnings.eq(''), 'No missing-field or training-date-range flags')

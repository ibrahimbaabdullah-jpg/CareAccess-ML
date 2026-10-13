"""Strict input schema. References must be nonidentifying temporary appointment tokens."""
import re
import numpy as np
import pandas as pd

INPUT_COLUMNS = ['appointment_id', 'appointment_datetime', 'specialty', 'age']
SPECIALTIES = ['speech therapy','psychotherapy','physiotherapy','occupational therapy','pedagogo','enf','assist','sem especialidade']
MAX_ROWS = 5000


def validate_frame(frame: pd.DataFrame):
    errors = []
    if not isinstance(frame, pd.DataFrame):
        raise ValueError('A CSV table is required.')
    if len(frame) > MAX_ROWS:
        return frame.iloc[:0].copy(), pd.DataFrame([{'row':0,'field':'file','message':f'Maximum {MAX_ROWS} rows.'}])
    missing = set(INPUT_COLUMNS) - set(frame.columns)
    extra = set(frame.columns) - set(INPUT_COLUMNS)
    if missing or extra or frame.columns.duplicated().any():
        msg=f'Required columns: {", ".join(INPUT_COLUMNS)}. Missing: {sorted(missing)}. Unsupported: {sorted(extra)}.'
        return pd.DataFrame(columns=INPUT_COLUMNS), pd.DataFrame([{'row':0,'field':'columns','message':msg}])
    data = frame.copy().reset_index(drop=True)
    invalid = set()
    def error(i, field, message):
        errors.append({'row':int(i)+1,'field':field,'message':message});invalid.add(i)
    refs=data.appointment_id.fillna('').astype(str)
    for i, value in refs.items():
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,39}', value):
            error(i,'appointment_id','Use a temporary nonidentifying reference (letters, digits, hyphens or underscores), never a patient identifier.')
    for i in data.index[refs.duplicated(keep=False)]: error(i,'appointment_id','Duplicate appointment reference.')
    dates=pd.to_datetime(data.appointment_datetime,format='%Y-%m-%dT%H:%M',errors='coerce')
    for i in data.index[dates.isna()]: error(i,'appointment_datetime','Use YYYY-MM-DDTHH:MM, local appointment time without timezone.')
    for i in data.index[dates.notna() & ~dates.dt.year.between(2000,2100)]: error(i,'appointment_datetime','Year must be between 2000 and 2100.')
    ages=pd.to_numeric(data.age,errors='coerce')
    blank=data.age.isna() | data.age.astype(str).str.strip().eq('')
    for i in data.index[(~blank & (ages.isna() | ~np.isfinite(ages))) | (ages.notna() & ~ages.between(0,110))]:
        error(i,'age','Age at appointment must be 0–110 or left blank.')
    specs=data.specialty.fillna('').astype(str).str.strip()
    for i in data.index[~specs.isin(SPECIALTIES+[''])]: error(i,'specialty','Unrecognized specialty; use a listed category or leave blank.')
    data['age']=ages
    data['specialty']=specs.replace('',np.nan)
    data['appointment_id']=refs
    return data.drop(index=list(invalid)).reset_index(drop=True), pd.DataFrame(errors,columns=['row','field','message'])


def make_features(frame):
    """Allowlist prevents target, reasons, observed weather and demographics entering ML."""
    date = pd.to_datetime(frame.appointment_datetime,format='%Y-%m-%dT%H:%M',errors='raise')
    return pd.DataFrame({'age':pd.to_numeric(frame.age,errors='coerce'), 'hour':date.dt.hour+date.dt.minute/60,
                         'weekday':date.dt.dayofweek, 'month':date.dt.month,
                         'specialty':frame.specialty.fillna('missing').astype(str)},index=frame.index)

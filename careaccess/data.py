"""Restricted raw data preparation. Never ship original records or birth dates."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

SOURCE_SHA256='903f30aa008c478dd6451550b53c2bf937653f28763ea288e43f5e99c536cbdf'


def prepare_source(path):
    path=Path(path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=SOURCE_SHA256: raise ValueError('Source checksum mismatch: expected verified Mendeley version 1.')
    raw=pd.read_csv(path)
    counts={'original_rows':len(raw),'original_columns':len(raw.columns),'exact_duplicate_rows_removed':int(raw.duplicated().sum())}
    data=raw.drop_duplicates().copy()
    date=pd.to_datetime(data.appointment_date,format='%d/%m/%Y',errors='coerce')
    birth=pd.to_datetime(data.date_of_birth,format='%d/%m/%Y',errors='coerce')
    age=(date-birth).dt.total_seconds()/(365.2425*86400)
    age=age.where(age.between(0,110))
    timestamp=pd.to_datetime(data.appointment_date+' '+data.appointment_time,format='%d/%m/%Y %H:%M',errors='coerce')
    valid=timestamp.notna() & data.no_show.isin(['yes','no'])
    counts['invalid_datetime_or_target_removed']=int((~valid).sum())
    frame=pd.DataFrame({'appointment_datetime':timestamp.dt.strftime('%Y-%m-%dT%H:%M'),'specialty':data.specialty,
        'age':age,'target':data.no_show.map({'yes':1,'no':0}),'gender_audit':data.gender})[valid].copy()
    frame=frame.sort_values('appointment_datetime',kind='stable').reset_index(drop=True)
    counts.update(clean_rows=len(frame),missing_derived_age=int(frame.age.isna().sum()),missing_specialty=int(frame.specialty.isna().sum()),missed=int(frame.target.sum()))
    return frame,counts


def temporal_split(frame):
    dates=pd.to_datetime(frame.appointment_datetime).dt.normalize()
    bins=[dates < '2019-01-01', (dates >= '2019-01-01') & (dates < '2020-01-01'),
          (dates >= '2020-01-01') & (dates < '2021-01-01'), dates >= '2021-01-01']
    return {name:frame.loc[mask].reset_index(drop=True) for name,mask in zip(['train','validation','calibration','test'],bins)}

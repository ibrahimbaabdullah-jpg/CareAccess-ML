import os

import numpy as np
import pandas as pd
import pytest

from careaccess.data import temporal_split
from careaccess.inference import load_bundle, model_names, predict
from careaccess.validation import INPUT_COLUMNS, make_features, validate_frame


@pytest.fixture(scope='module')
def bundle(): return load_bundle()
@pytest.fixture
def example():
    path=os.environ.get('CAREACCESS_TEST_DATA')
    if not path:pytest.skip('Private actual-source fixture not provided; no substitute records generated.')
    return pd.read_csv(path)

def test_examples_valid(example):
    clean,errors=validate_frame(example);assert errors.empty;assert len(clean)==24

@pytest.mark.parametrize('field,value',[('age',-1),('age',111),('age','bad'),('age',float('inf')),('specialty','made up'),('appointment_datetime','03/04/2025'),('appointment_datetime','2025-02-31T10:00'),('appointment_id','patient real name')])
def test_bad_rows_rejected(example,field,value):
    example[field]=example[field].astype(object);example.loc[0,field]=value
    clean,errors=validate_frame(example);assert len(clean)==23;assert len(errors)>0

@pytest.mark.parametrize('column',['no_show','no_show_reason','average_temp_day','date_of_birth','patient_name','gender','icd'])
def test_forbidden_columns_rejected(example,column):
    example[column]='anything';clean,errors=validate_frame(example);assert clean.empty and not errors.empty

def test_duplicate_ref(example):
    example.loc[1,'appointment_id']=example.loc[0,'appointment_id'];clean,errors=validate_frame(example);assert len(clean)==22

def test_missing_imputed_and_no_leakage(example,bundle):
    example['age']=np.nan;example['specialty']=np.nan
    for n in model_names(bundle):assert np.isfinite(predict(example,n,bundle)).all()
    x=make_features(example);assert list(x)==['age','hour','weekday','month','specialty']

@pytest.mark.parametrize('name',['Constant baseline','Logistic Regression','Gradient Boosting','PyTorch Neural Network'])
def test_actual_inference_repeatable(example,bundle,name):
    p=predict(example,name,bundle);q=predict(example,name,bundle)
    assert p.shape==(24,) and np.all((p>=0)&(p<=1));np.testing.assert_allclose(p,q,rtol=0,atol=1e-8)
    if name!='Constant baseline':assert np.ptp(p)>1e-5

def test_single_batch_match(example,bundle):
    for name in model_names(bundle):
        full=predict(example,name,bundle);single=np.concatenate([predict(example.iloc[[i]],name,bundle) for i in range(len(example))]);np.testing.assert_allclose(full,single,rtol=0,atol=1e-6)

def test_empty_and_invalid_model(example,bundle):
    assert len(predict(example.iloc[:0],bundle=bundle))==0
    with pytest.raises(ValueError):predict(example,'imaginary',bundle)

def test_split_boundaries():
    x=pd.DataFrame({'appointment_datetime':['2018-12-31T10:00','2019-01-01T10:00','2020-01-01T10:00','2021-01-01T10:00']})
    parts=temporal_split(x);assert all(len(v)==1 for v in parts.values())

def test_saved_reload_parity(example,bundle):
    reloaded=load_bundle()
    for name in model_names(bundle):np.testing.assert_allclose(predict(example,name,bundle),predict(example,name,reloaded),atol=1e-6,rtol=0)

def test_frozen_report_consistent(bundle):
    r=bundle['report'];assert r['counts']['clean_rows']==sum(v['rows'] for v in r['splits'].values())==46672
    assert r['splits']['test']['rows']==8905
    for m in r['metrics']:assert np.array(m['confusion_matrix']).sum()==8905

@pytest.mark.parametrize('name',['Constant baseline','Logistic Regression','Gradient Boosting','PyTorch Neural Network'])
def test_model_integrity_without_patient_fixture(bundle,name):
    import torch
    model=bundle['models'][name]
    if isinstance(model,torch.nn.Module):
        weights=list(model.parameters());assert sum(w.numel() for w in weights)>100
        assert all(torch.isfinite(w).all() for w in weights)
        assert not model.training
    else:
        assert model.n_features_in_==bundle['manifest']['input_dim']
        assert list(model.classes_)==[0,1]

def test_empty_public_template():
    template=pd.read_csv('examples/appointments_template.csv')
    assert template.empty and list(template)==INPUT_COLUMNS

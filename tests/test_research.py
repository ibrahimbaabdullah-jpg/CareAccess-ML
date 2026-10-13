"""Read-only research evidence and cohort-separation contracts; no invented patients."""
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
from careaccess.inference import load_bundle, model_names
from careaccess.research import SUMMARY_PATH, SUMMARY_SHA256, load_research_summary, policy_table
ROOT = Path(__file__).resolve().parents[1]

def research_page():
    at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=60).run()
    at.sidebar.radio[0].set_value('Research / Vitória').run()
    assert not at.exception
    return at

def visible_text(at):
    return '\n'.join(str(item.value) for kind in ['markdown','caption','warning','info','error'] for item in at.get(kind))

def test_frozen_summary_integrity_and_cohort_separation():
    d = load_research_summary()
    assert sha256(SUMMARY_PATH.read_bytes()).hexdigest() == SUMMARY_SHA256
    assert d['deployment_changed'] is False and d['selected_id'] == 'xgb_d3'
    assert d['locked_rule'] == 'location_lead_prior'
    assert d['gates']['C'] == 'PASS' and d['gates']['D'] == 'FAIL'
    assert d['cohort']['appointments'] == 15016
    assert d['cohort']['events'] == 3906 and d['cohort']['slots'] == 1498
    assert d['cohort']['observed_dates'] == 8 and d['cohort']['observed_iso_weeks'] == 2
    assert d['design']['learned_fits'] == 40
    assert d['primary_probability_metrics']['recalibration_applied'] is False

def test_summary_is_bounded_cohort_aggregates_only():
    d = load_research_summary()
    assert SUMMARY_PATH.stat().st_size < 32768
    prohibited = {'PatientId','AppointmentID','appointment_id','patient_id','patient_key','row_hash','scores','weights','thresholds','bins','subgroups','by_date','by_iso_week','selections','records'}
    def check(value):
        if isinstance(value,dict):
            assert not prohibited.intersection(value)
            for nested in value.values(): check(nested)
        elif isinstance(value,list):
            assert len(value) <= 10
            for nested in value: check(nested)
    check(d)
    assert len(d['policies']) == 10 and len(d['comparisons']) == 3
    assert d['verification']['sanitized_aggregate_sha256'] == '064a1d2b4808f609e20f7607d2f868e45f4222d50469893704452f0183514ad2'

def test_all_policy_metrics_reconcile_without_reselecting():
    d = load_research_summary(); rows = {p['id']:p for p in d['policies']}
    assert rows['xgb_d3']['captured_events'] == 544
    assert rows['location_lead_prior']['captured_events'] == 460
    assert rows['lr_c1']['captured_events'] == 567
    assert rows['matched_random']['capture_is_expectation'] is True
    for p in rows.values():
        assert p['precision'] == pytest.approx(p['captured_events']/1498)
        assert p['capture_fraction'] == pytest.approx(p['captured_events']/3906)
    for c in d['comparisons']:
        difference = 544-rows[c['comparator']]['captured_events']
        assert c['capture_difference'] == pytest.approx(difference)
        assert c['precision_difference'] == pytest.approx(difference/1498)
    assert d['selected_id'] != max(rows,key=lambda name:rows[name]['captured_events'])

def test_summary_intervals_and_overprediction_preserved():
    d=load_research_summary();rule,logistic,random=d['comparisons']
    assert rule['date_ci95']['lower'] == pytest.approx(.026826734014266433)
    assert rule['crossed_patient_date_ci95']['lower'] == pytest.approx(.006226691106573512)
    assert logistic['date_ci95']['lower'] < 0 < logistic['date_ci95']['upper']
    assert logistic['crossed_patient_date_ci95']['lower'] < 0 < logistic['crossed_patient_date_ci95']['upper']
    p=d['primary_probability_metrics']
    assert p['observed_expected_ratio'] == pytest.approx(p['observed_events']/p['expected_events'])
    assert p['observed_expected_ratio'] < 1
    assert 'coverage guarantee' in d['uncertainty']['limits']

@pytest.mark.parametrize('payload',[b'',b'{}',b'null',b'['*32769])
def test_changed_or_oversized_summary_rejected(tmp_path,payload):
    path=tmp_path/'summary.json';path.write_bytes(payload)
    with pytest.raises(ValueError,match='integrity'):load_research_summary(path)

def test_missing_summary_rejected(tmp_path):
    with pytest.raises(OSError):load_research_summary(tmp_path/'absent.json')

def test_table_labels_random_as_expectation_and_preserves_order():
    table=policy_table(load_research_summary(),{'xgb_d3','location_lead_prior','lr_c1','matched_random'})
    assert table['Recorded no-shows identified'].tolist() == ['544','460','567','389.67 expected']
    assert table['Precision in selected slots'].tolist() == ['36.32%','30.71%','37.85%','26.01%']

def test_research_page_is_read_only_and_original_warning_remains():
    at=research_page();assert not at.error;text=visible_text(at)
    for phrase in ['0.505','close to chance','does not power Live Prediction','not pooled','No model was reselected','Raw probabilities overpredicted','younger adult','eight appointment dates favored the rule','coverage guarantee','CC BY-NC-SA 4.0','Consent, institutional release authority']:
        assert phrase in text
    assert [item.value for item in at.metric][:3] == ['544 / 1,498','PASS','FAIL']
    assert not at.get('file_uploader') and not at.number_input and not at.selectbox
    assert 'batch_scores' not in at.session_state and 'last_single_prediction' not in at.session_state

def test_download_exactly_matches_verified_aggregate_evidence():
    at=research_page();downloads={};original=st.download_button
    def capture(label,data,*args,**kwargs):
        downloads[label]=data
        return original(label,data,*args,**kwargs)
    with patch('streamlit.download_button',side_effect=capture):at.run()
    assert not at.exception
    assert list(downloads) == ['Download aggregate research summary']
    assert json.loads(downloads['Download aggregate research summary']) == load_research_summary()

@pytest.mark.parametrize('error',[ValueError('bad checksum'),FileNotFoundError('missing')])
def test_unavailable_research_does_not_offer_unverified_download_or_break_prediction(error):
    at=research_page()
    with patch('careaccess.research.load_research_summary',side_effect=error):at.run()
    assert not at.exception
    assert any('verified research summary is unavailable' in item.value for item in at.error)
    assert not at.metric and not at.dataframe and not at.get('download_button')
    at.sidebar.radio[0].set_value('Live Prediction').run()
    assert not at.exception and not at.error and at.date_input[0].value is None
    assert at.selectbox(key='live_model').value == load_bundle()['default_model']

def test_repeated_research_navigation_keeps_original_model_choices_and_blank_input():
    at=research_page()
    for _ in range(2):
        at.sidebar.radio[0].set_value('Live Prediction').run()
        assert set(at.selectbox(key='live_model').options) == set(model_names(load_bundle()))
        assert not any('XGBoost' in value or 'Vitória' in value for value in at.selectbox(key='live_model').options)
        assert at.date_input[0].value is None and at.time_input[0].value is None
        at.sidebar.radio[0].set_value('Research / Vitória').run()
        assert not at.exception

def test_research_navigation_preserves_consent_without_creating_scores():
    at=research_page();at.sidebar.radio[0].set_value('Batch Prediction').run()
    at.checkbox(key='batch_confirm').check().run()
    at.sidebar.radio[0].set_value('Research / Vitória').run()
    at.sidebar.radio[0].set_value('Batch Prediction').run()
    assert at.checkbox(key='batch_confirm').value and 'batch_scores' not in at.session_state
    at.checkbox(key='batch_confirm').uncheck().run()
    at.sidebar.radio[0].set_value('Research / Vitória').run()
    at.sidebar.radio[0].set_value('Batch Prediction').run()
    assert not at.checkbox(key='batch_confirm').value

@pytest.mark.parametrize('destination',['Live Prediction','Batch Prediction','About / Documentation'])
def test_hosted_input_boundary_is_explicit(destination):
    at=research_page();at.sidebar.radio[0].set_value(destination).run()
    text=visible_text(at)
    for phrase in ['Do not upload patient records','Public availability, removing names, or checking a consent box','separately governed deployment']:
        assert phrase in text
    assert not at.exception

def test_private_offline_scores_capacity_and_reviews_survive_research_navigation():
    import os
    import pandas as pd
    from careaccess.inference import predict
    path=os.environ.get('CAREACCESS_TEST_DATA')
    if not path or not Path(path).is_file():
        pytest.skip('Private real-source fixture required; no synthetic appointments substituted')
    inputs=pd.read_csv(path,dtype={'appointment_id':str})
    scores=inputs.copy();scores['miss_probability']=predict(inputs)
    at=research_page()
    at.session_state['batch_scores']=scores
    at.session_state['batch_authorized']=True
    at.session_state['batch_capacity']=5
    at.session_state['batch_review_statuses']={inputs.appointment_id.iloc[0]:'Needs further review'}
    for _ in range(2):
        at.sidebar.radio[0].set_value('Outreach Planner').run()
        assert at.number_input[0].value == 5 and at.metric[0].value == '5'
        at.sidebar.radio[0].set_value('Research / Vitória').run()
        pd.testing.assert_frame_equal(at.session_state['batch_scores'],scores)
        assert at.session_state['batch_review_statuses'][inputs.appointment_id.iloc[0]] == 'Needs further review'
        assert not at.exception

from careaccess.session import (clear_batch_session, reset_batch_results, revoke_batch_authorization,
                                prepare_batch_authorization, prepare_planner_capacity, remember_planner_capacity)


def test_clear_drops_upload_and_all_batch_results_and_resets_confirmation():
    state = {
        'batch_scores': object(), 'batch_upload_0': object(),
        'batch_confirm': True, 'batch_review_statuses': {'old': 'reviewed'},
        'batch_review_editor': {}, 'live_model': 'Logistic Regression',
    }
    clear_batch_session(state)
    assert state == {
        'batch_upload_generation': 1, 'batch_confirm': False, 'batch_authorized': False,
        'live_model': 'Logistic Regression',
    }
    clear_batch_session(state)
    assert state['batch_upload_generation'] == 2


def test_new_scoring_attempt_removes_prior_reviews():
    state = {'batch_scores': object(), 'batch_review_statuses': {},
             'batch_review_editor_0_5': {}, 'batch_upload_0': object()}
    reset_batch_results(state)
    assert set(state) == {'batch_result_generation', 'batch_upload_0'}
    assert state['batch_result_generation'] == 1


def test_authorization_revocation_clears_all_batch_state():
    state = {'batch_confirm': False, 'batch_scores': object(), 'batch_upload_0': object()}
    revoke_batch_authorization(state)
    assert state == {'batch_confirm': False, 'batch_upload_generation': 1, 'batch_authorized': False}


def test_granting_authorization_keeps_empty_upload_slot():
    state = {'batch_confirm': True, 'batch_upload_generation': 1}
    revoke_batch_authorization(state)
    assert state == {'batch_confirm': True, 'batch_upload_generation': 1, 'batch_authorized': True}


def test_authorization_survives_widget_cleanup_until_explicitly_revoked():
    state = {'batch_confirm': True}
    revoke_batch_authorization(state)
    del state['batch_confirm']  # Streamlit removes off-page widget state.
    prepare_batch_authorization(state)
    assert state['batch_confirm'] is True
    state['batch_confirm'] = False
    revoke_batch_authorization(state)
    prepare_batch_authorization(state)
    assert state['batch_confirm'] is False
    assert state['batch_authorized'] is False


def test_no_default_or_inferred_authorization():
    state = {}
    prepare_batch_authorization(state)
    assert state['batch_confirm'] is False


def test_legacy_results_without_durable_consent_are_cleared():
    state = {'batch_scores': object(), 'batch_capacity': 5}
    prepare_batch_authorization(state)
    assert state['batch_confirm'] is False
    assert 'batch_scores' not in state
    assert 'batch_capacity' not in state


def test_capacity_survives_widget_cleanup_and_clamps_to_new_batch():
    state = {}
    prepare_planner_capacity(state, 24)
    assert state['batch_capacity_input'] == 20
    state['batch_capacity_input'] = 5
    remember_planner_capacity(state)
    del state['batch_capacity_input']
    prepare_planner_capacity(state, 24)
    assert state['batch_capacity_input'] == 5
    prepare_planner_capacity(state, 3)
    assert state['batch_capacity_input'] == state['batch_capacity'] == 3
    prepare_planner_capacity(state, 0)
    assert state['batch_capacity_input'] == state['batch_capacity'] == 0


def test_clear_resets_durable_capacity_and_authorization():
    state = {'batch_authorized': True, 'batch_confirm': True,
             'batch_capacity': 5, 'batch_capacity_input': 5}
    clear_batch_session(state)
    assert not state['batch_authorized']
    assert 'batch_capacity' not in state and 'batch_capacity_input' not in state
    prepare_planner_capacity(state, 24)
    assert state['batch_capacity_input'] == 20

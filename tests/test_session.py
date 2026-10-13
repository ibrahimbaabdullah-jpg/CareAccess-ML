from careaccess.session import clear_batch_session, reset_batch_results, revoke_batch_authorization


def test_clear_drops_upload_and_all_batch_results_and_resets_confirmation():
    state = {
        'batch_scores': object(), 'batch_upload_0': object(),
        'batch_confirm': True, 'batch_review_statuses': {'old': 'reviewed'},
        'batch_review_editor': {}, 'live_model': 'Logistic Regression',
    }
    clear_batch_session(state)
    assert state == {
        'batch_upload_generation': 1, 'batch_confirm': False,
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
    assert state == {'batch_confirm': False, 'batch_upload_generation': 1}


def test_granting_authorization_keeps_empty_upload_slot():
    state = {'batch_confirm': True, 'batch_upload_generation': 1}
    revoke_batch_authorization(state)
    assert state == {'batch_confirm': True, 'batch_upload_generation': 1}

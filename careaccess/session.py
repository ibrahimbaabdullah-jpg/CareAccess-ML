"""Explicit session lifecycle for uploaded schedules and their derived results."""


def reset_batch_results(state):
    """Start a new scoring attempt without carrying forward human reviews."""
    for key in list(state):
        if key.startswith("batch_review_") or key == "batch_scores":
            state.pop(key, None)
    state["batch_result_generation"] = state.get("batch_result_generation", 0) + 1


def clear_batch_session(state):
    """Drop this app's batch state and rotate the upload widget on the next run.

    This clears the active application session, not hosting-provider records or
    copies already downloaded by the user.
    """
    generation = state.get("batch_upload_generation", 0)
    for key in list(state):
        if key.startswith(("batch_upload_", "batch_review_")) or key in {
            "batch_scores", "batch_result_generation"
        }:
            state.pop(key, None)
    state["batch_upload_generation"] = generation + 1
    state["batch_confirm"] = False


def revoke_batch_authorization(state):
    if not state.get("batch_confirm", False):
        clear_batch_session(state)

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
            "batch_scores", "batch_result_generation", "batch_capacity", "batch_capacity_input"
        }:
            state.pop(key, None)
    state["batch_upload_generation"] = generation + 1
    state["batch_confirm"] = False
    state["batch_authorized"] = False


def revoke_batch_authorization(state):
    state["batch_authorized"] = bool(state.get("batch_confirm", False))
    if not state["batch_authorized"]:
        clear_batch_session(state)


def prepare_batch_authorization(state):
    """Restore the consent widget from durable, explicitly granted consent.

    Streamlit removes off-page widget values; that must not look like withdrawal
    while the same authorized batch and its exports remain in the session.
    """
    authorized = bool(state.get("batch_authorized", False))
    if not authorized and "batch_scores" in state:
        # Fail closed for a legacy/inconsistent session without durable consent.
        clear_batch_session(state)
    state["batch_confirm"] = authorized


def prepare_planner_capacity(state, row_count):
    """Restore the capacity widget, clamping it when a replacement batch shrinks."""
    capacity = min(max(int(state.get("batch_capacity", 20)), 0), row_count)
    state["batch_capacity"] = capacity
    state["batch_capacity_input"] = capacity


def remember_planner_capacity(state):
    state["batch_capacity"] = int(state["batch_capacity_input"])

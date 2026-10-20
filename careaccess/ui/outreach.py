"""Capacity-limited outreach review list built from the last scored batch."""

import streamlit as st

from careaccess.planner import export_csv, plan_outreach
from careaccess.session import (
    prepare_planner_capacity,
    remember_planner_capacity,
)


def render_planner(bundle):
    st.header("Create a supportive outreach review list")
    st.info(
        "Ranking estimates non-attendance risk, not who benefits most from a call. Keep ordinary reminders and access to care available to everyone."
    )
    scores = st.session_state.get("batch_scores")
    if scores is None:
        st.write(
            "First open Batch Prediction and score an authorized, non-identifying schedule. Your results will appear here in this session."
        )
    else:
        st.caption(f"Using the last successfully scored schedule: {len(scores):,} appointments.")
        st.caption(
            "Capacity is the total number of appointments across this entire uploaded batch, including all appointment dates. Equal scores are ordered by appointment reference."
        )
        prepare_planner_capacity(st.session_state, len(scores))
        capacity = st.number_input(
            "How many appointments can your team contact?",
            min_value=0,
            max_value=len(scores),
            step=1,
            key="batch_capacity_input",
            on_change=remember_planner_capacity,
            args=(st.session_state,),
        )
        planned = plan_outreach(scores, int(capacity))
        st.metric("Appointments selected for review", len(planned))
        if not planned.empty:
            planned = planned.copy()
            statuses = st.session_state.setdefault("batch_review_statuses", {})
            planned["review_status"] = planned.appointment_id.map(statuses).fillna("Not reviewed")
            result_generation = st.session_state.get("batch_result_generation", 0)
            planned = st.data_editor(
                planned,
                key=f"batch_review_editor_{result_generation}_{int(capacity)}",
                hide_index=True,
                width="stretch",
                disabled=[c for c in planned.columns if c != "review_status"],
                column_config={
                    "review_status": st.column_config.SelectboxColumn(
                        "Review status",
                        options=[
                            "Not reviewed",
                            "Ready for supportive outreach",
                            "Needs further review",
                            "Reviewed — no additional outreach",
                        ],
                    ),
                    "miss_probability": st.column_config.NumberColumn(
                        "Estimated non-attendance probability", format="%.4f"
                    ),
                },
            )
            statuses.update(zip(planned.appointment_id, planned.review_status, strict=True))
        else:
            st.info("Capacity is zero. No appointments are selected.")
        st.download_button(
            "Download outreach review list", export_csv(planned), "careaccess_outreach_review.csv", "text/csv"
        )
        st.caption(
            "A human should review the list and consider practical barriers, patient preferences and local policy. The app does not send messages, place calls or change appointments."
        )

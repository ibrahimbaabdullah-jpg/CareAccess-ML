"""Live and batch prediction pages."""


import pandas as pd
import streamlit as st

from careaccess.csv_input import read_appointments_csv
from careaccess.planner import export_csv
from careaccess.session import (
    clear_batch_session,
    reset_batch_results,
    revoke_batch_authorization,
)
from careaccess.ui.common import (
    HOST_INPUT_NOTICE,
    empty_template,
    label,
    model_picker,
    show_scores,
    validate_and_score,
)


def render_live(bundle):
    contract = bundle["feature_contract"]
    st.header("Enter an appointment")
    st.warning(HOST_INPUT_NOTICE)
    st.write("Use only permitted nonsensitive, non-identifying inputs. The app runs the selected saved model on your inputs; no appointment is prefilled.")
    model = model_picker(bundle, "live_model")
    with st.form("single_appointment"):
        left, right = st.columns(2)
        appointment_date = left.date_input("Appointment date", value=None)
        appointment_time = right.time_input("Appointment time (clinic local time)", value=None)
        age = left.number_input("Age at appointment", min_value=0.0, max_value=110.0, value=None, step=1.0)
        specialties = contract.get("specialty_values", [])
        specialty = right.selectbox("Specialty (optional)", specialties, index=None, placeholder="Choose specialty, or leave blank") if specialties else right.text_input("Specialty (optional)", value="")
        confirmed = st.checkbox("I confirm these inputs are permitted on this host, contain no sensitive or patient-level health information, and do not identify anyone.", key="live_confirm")
        submitted = st.form_submit_button("Estimate non-attendance probability", type="primary")
    if submitted:
        if not confirmed:
            st.warning("Please confirm authorization and privacy before continuing.")
        elif appointment_date is None or appointment_time is None:
            st.error("Enter an appointment date and time before scoring.")
        else:
            frame = pd.DataFrame([{"appointment_id": "ROW-1", "appointment_datetime": f"{appointment_date.isoformat()}T{appointment_time:%H:%M}", "age": age, "specialty": specialty}])
            try:
                scored = validate_and_score(frame, model, bundle)
                if scored is not None:
                    st.session_state["last_single_prediction"] = scored.to_dict("records")[0]
                    st.metric("Estimated non-attendance probability", f"{scored['miss_probability'].iloc[0]:.1%}")
                    st.caption(f"Computed with {label(model)} using the appointment entered above.")
                    show_scores(scored)
                    st.download_button("Download this prediction", export_csv(scored), "careaccess_single_prediction.csv", "text/csv")
            except ValueError as error:
                st.error(str(error))
    with st.expander("How should I interpret this estimate?"):
        st.write("The model uses patterns learned from historical appointments. It does not know why someone might miss a visit, whether a reminder will help, or their current circumstances. Review input quality and local applicability before interpreting a score.")
        st.write("Feature importance in Model Evaluation describes overall model behavior; it is not a causal explanation of an individual prediction.")


def render_batch(bundle):
    st.header("Validate and score a schedule")
    st.warning(HOST_INPUT_NOTICE)
    st.write("Only permitted nonsensitive, non-identifying inputs may be processed here. Files are processed in session memory and are not deliberately saved to disk or used for training.")
    st.download_button("Download empty CSV template", export_csv(empty_template()), "careaccess_empty_template.csv", "text/csv")
    st.caption("The template contains column names only. Dates must use YYYY-MM-DDTHH:MM. Appointment references must not identify patients; use local row references. Age and specialty may be left blank.")
    confirmed = st.checkbox("I confirm this file is permitted on this host, contains no sensitive or patient-level health information, and does not identify anyone.", key="batch_confirm", on_change=revoke_batch_authorization, args=(st.session_state,))
    frame = None
    uploaded = None
    if confirmed:
        generation = st.session_state.get("batch_upload_generation", 0)
        uploaded = st.file_uploader("Appointments CSV (maximum 5 MB / 5,000 rows)", type=["csv"], key=f"batch_upload_{generation}")
        if uploaded is not None:
            try:
                frame = read_appointments_csv(uploaded.getvalue())
            except ValueError as error:
                reset_batch_results(st.session_state)
                st.error(str(error))
    if frame is not None:
        st.caption(f"{len(frame):,} appointment rows")
        st.dataframe(frame.head(50), hide_index=True, width="stretch")
        model = model_picker(bundle, "batch_model")
        if st.button("Validate and score appointments", type="primary"):
            reset_batch_results(st.session_state)
            if not confirmed:
                st.warning("Please confirm authorization and privacy before scoring.")
            else:
                try:
                    scored = validate_and_score(frame, model, bundle)
                    if scored is not None:
                        st.session_state["batch_scores"] = scored
                        st.success(f"Scored {len(scored):,} appointments. Open Outreach Planner to create a review list.")
                except ValueError as error:
                    st.error(str(error))
    if "batch_scores" in st.session_state:
        st.subheader("Last successfully scored schedule")
        st.caption("These results are from the last scoring run. Changing the file or model above does not update them until you score again.")
        show_scores(st.session_state["batch_scores"])
        st.download_button("Download scored schedule", export_csv(st.session_state["batch_scores"]), "careaccess_predictions.csv", "text/csv")
    if uploaded is not None or "batch_scores" in st.session_state:
        st.button("Clear schedule and results", on_click=clear_batch_session, args=(st.session_state,))
        st.caption("Clears this session's uploaded schedule and batch results. It does not erase downloaded copies or hosting-provider records.")

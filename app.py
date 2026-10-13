"""CareAccess ML: a privacy-aware public interface to saved trained models."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st

from careaccess.inference import load_bundle, predict, model_names
from careaccess.validation import validate_frame
from careaccess.planner import plan_outreach, export_csv
from careaccess.csv_input import read_appointments_csv
from careaccess.session import clear_batch_session, reset_batch_results, revoke_batch_authorization
from careaccess.quality import input_quality_warnings

ROOT = Path(__file__).resolve().parent
PAGES = ["Home / Dashboard", "Live Prediction", "Batch Prediction", "Outreach Planner", "Model Comparison", "Model Evaluation", "About / Documentation"]
LABELS = {"logistic": "Logistic Regression", "logistic_regression": "Logistic Regression", "gradient_boosting": "Gradient Boosting", "hist_gradient_boosting": "Gradient Boosting", "pytorch": "PyTorch Neural Network", "neural_network": "PyTorch Neural Network", "baseline": "Constant baseline"}

st.set_page_config(page_title="CareAccess ML", page_icon="📅", layout="wide")
st.markdown("""<style>
.block-container {max-width: 1200px; padding-top: 2rem;}
[data-testid="stMetric"] {background:#edf4f6; padding:1rem; border-radius:12px;}
h1,h2,h3 {letter-spacing:-.025em;}
@media(max-width:640px){.block-container{padding-left:1rem;padding-right:1rem;}}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def trained_bundle():
    return load_bundle()


def label(name):
    return LABELS.get(name, str(name) if " " in str(name) else str(name).replace("_", " ").title())


def model_picker(bundle, key):
    names = list(model_names(bundle))
    default = bundle["default_model"]
    return st.selectbox("Trained model", names, index=names.index(default) if default in names else 0, format_func=label, key=key)


def empty_template():
    return pd.DataFrame(columns=["appointment_id", "appointment_datetime", "specialty", "age"])


def validate_and_score(frame, model, bundle):
    clean, errors = validate_frame(frame)
    if not errors.empty:
        st.error("Please correct these problems before scoring. No appointments have been scored.")
        st.dataframe(errors, hide_index=True, width="stretch")
        return None
    probabilities = predict(clean, model_name=model, bundle=bundle)
    result = clean.copy()
    result["miss_probability"] = probabilities
    result["model"] = model
    result["input_quality_warnings"] = input_quality_warnings(clean, bundle["feature_contract"])
    return result


def show_scores(frame):
    st.dataframe(frame, hide_index=True, width="stretch", column_config={"miss_probability": st.column_config.NumberColumn("Estimated non-attendance probability", format="%.4f")})
    st.caption("Probabilities are between 0 and 1: 0.30 means an estimated 30%. These are uncertain estimates from historical data, not a statement about a person's intentions.")


def metric_rows(report):
    metrics = report.get("metrics", {})
    if isinstance(metrics, list):
        return pd.DataFrame(metrics)
    return pd.DataFrame([{"model": name, **values} for name, values in metrics.items()])


st.sidebar.title("CareAccess ML")
st.sidebar.caption("Appointment outreach planning")
page = st.sidebar.radio("Explore", PAGES)
st.sidebar.info("Use only non-identifying data you are authorized to process. Do not enter names, contact details, medical record numbers or confidential patient data.")
st.sidebar.caption("Supportive reminders only. Never cancel appointments, restrict care, or penalize patients using a score.")
st.title("CareAccess ML")
st.caption("Missed-Appointment Prediction & Outreach Planner")

try:
    bundle = trained_bundle()
except Exception:
    st.error("The verified model bundle could not be loaded. Predictions are unavailable. Please ask the maintainer to check the model files and runtime logs.")
    st.stop()
report = bundle["report"]
contract = bundle["feature_contract"]
manifest = bundle["manifest"]
default_test_result = next((row for row in report.get("metrics", []) if row.get("model") == bundle["default_model"]), {})
if default_test_result.get("roc_auc", 1.0) < 0.60:
    st.warning(f"Important result: the default model's final-test ROC-AUC is {default_test_result['roc_auc']:.3f}, close to chance (0.500). This release demonstrates working trained-model inference, but does not establish effective outreach targeting. Do not use its rankings for real patient decisions.")

if page == "Home / Dashboard":
    st.header("Plan supportive outreach, one appointment at a time")
    st.write("Explore real trained models that estimate missed-appointment probability, then build a review list that fits your team's reminder-call capacity.")
    st.info("Model training and evaluation use the documented historical Mendeley dataset. No generated appointment cohort is supplied. Prediction performance does not prove that outreach prevents missed visits.")
    st.subheader("Three simple steps")
    cols = st.columns(3)
    for col, title, body in zip(cols, ["1 · Enter an appointment", "2 · Validate a schedule", "3 · Choose your outreach capacity"], ["Open Live Prediction to run a saved trained model.", "Use Batch Prediction to check an authorized, non-identifying CSV.", "Open Outreach Planner to create and download a ranked review list."]):
        col.markdown(f"**{title}**")
        col.write(body)
    counts = report.get("counts", {})
    counts_cols = st.columns(3)
    counts_cols[0].metric("Source appointment records", f"{counts['original_rows']:,}" if "original_rows" in counts else "Not available")
    counts_cols[1].metric("Records after documented cleaning", f"{counts['clean_rows']:,}" if "clean_rows" in counts else "Not available")
    counts_cols[2].metric("Default trained model", label(bundle["default_model"]))
    st.caption("These are appointment records, not a count of unique patients.")
    st.subheader("Measured model performance")
    rows = metric_rows(report)
    if not rows.empty:
        st.dataframe(rows, hide_index=True, width="stretch")
    else:
        st.warning("Evaluation metrics are not available in this model bundle.")
    st.write(f"Default model: **{label(bundle['default_model'])}**")
    st.caption("See Model Comparison and Model Evaluation for test design, thresholds, probability reliability and limitations.")
    st.download_button("Download empty CSV template", export_csv(empty_template()), "careaccess_empty_template.csv", "text/csv")

elif page == "Live Prediction":
    st.header("Enter an appointment")
    st.write("Enter only non-identifying information you are authorized to process. The app runs the selected saved model on your inputs; no appointment is prefilled.")
    model = model_picker(bundle, "live_model")
    with st.form("single_appointment"):
        left, right = st.columns(2)
        appointment_date = left.date_input("Appointment date", value=None)
        appointment_time = right.time_input("Appointment time (clinic local time)", value=None)
        age = left.number_input("Age at appointment", min_value=0.0, max_value=110.0, value=None, step=1.0)
        specialties = contract.get("specialty_values", [])
        specialty = right.selectbox("Specialty (optional)", specialties, index=None, placeholder="Choose specialty, or leave blank") if specialties else right.text_input("Specialty (optional)", value="")
        confirmed = st.checkbox("I am authorized to process these inputs and have excluded identifying or confidential patient information.", key="live_confirm")
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

elif page == "Batch Prediction":
    st.header("Validate and score a schedule")
    st.write("Upload only non-identifying appointment data that you are authorized to process on this hosting service. Do not include patient names, contact details, medical record numbers or confidential information. Files are processed in session memory and are not deliberately saved to disk or used for training.")
    st.download_button("Download empty CSV template", export_csv(empty_template()), "careaccess_empty_template.csv", "text/csv")
    st.caption("The template contains column names only. Dates must use YYYY-MM-DDTHH:MM. Appointment references must not identify patients; use local row references. Age and specialty may be left blank.")
    confirmed = st.checkbox("I am authorized to process this file here and have excluded identifying or confidential patient information.", key="batch_confirm", on_change=revoke_batch_authorization, args=(st.session_state,))
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

elif page == "Outreach Planner":
    st.header("Create a supportive outreach review list")
    st.info("Ranking estimates non-attendance risk, not who benefits most from a call. Keep ordinary reminders and access to care available to everyone.")
    scores = st.session_state.get("batch_scores")
    if scores is None:
        st.write("First open Batch Prediction and score an authorized, non-identifying schedule. Your results will appear here in this session.")
    else:
        st.caption(f"Using the last successfully scored schedule: {len(scores):,} appointments.")
        st.caption("Capacity is the total number of appointments across this entire uploaded batch, including all appointment dates. Equal scores are ordered by appointment reference.")
        capacity = st.number_input("How many appointments can your team contact?", min_value=0, max_value=len(scores), value=min(20, len(scores)), step=1)
        planned = plan_outreach(scores, int(capacity))
        st.metric("Appointments selected for review", len(planned))
        if not planned.empty:
            planned = planned.copy()
            statuses = st.session_state.setdefault("batch_review_statuses", {})
            planned["review_status"] = planned.appointment_id.map(statuses).fillna("Not reviewed")
            result_generation = st.session_state.get("batch_result_generation", 0)
            planned = st.data_editor(planned, key=f"batch_review_editor_{result_generation}_{int(capacity)}", hide_index=True, width="stretch", disabled=[c for c in planned.columns if c != "review_status"], column_config={"review_status": st.column_config.SelectboxColumn("Review status", options=["Not reviewed", "Ready for supportive outreach", "Needs further review", "Reviewed — no additional outreach"]), "miss_probability": st.column_config.NumberColumn("Estimated non-attendance probability", format="%.4f")})
            statuses.update(zip(planned.appointment_id, planned.review_status))
        else:
            st.info("Capacity is zero. No appointments are selected.")
        st.download_button("Download outreach review list", export_csv(planned), "careaccess_outreach_review.csv", "text/csv")
        st.caption("A human should review the list and consider practical barriers, patient preferences and local policy. The app does not send messages, place calls or change appointments.")

elif page == "Model Comparison":
    st.header("Compare the trained models")
    st.write("All candidate models are shown, including the simple baseline. A neural network is not assumed to perform best.")
    rows = metric_rows(report)
    st.dataframe(rows, hide_index=True, width="stretch")
    st.subheader("Default model")
    st.write(label(bundle["default_model"]))
    st.write(report.get("selection", "The default is selected using validation results, not the held-out final test. See the reproducible evaluation report for selection details."))
    if "model" in rows.columns:
        numeric = [c for c in ["roc_auc", "average_precision", "brier"] if c in rows.columns]
        if numeric:
            st.bar_chart(rows.set_index("model")[numeric])
    st.caption("Higher ROC-AUC and average precision are better; lower Brier score is better. Compare probabilities and operational capacity metrics, not accuracy alone.")

elif page == "Model Evaluation":
    st.header("Inspect the evidence and limitations")
    st.write("These measured results come from held-out historical Mendeley appointments. Interactive inputs and uploaded files do not change these evaluation results.")
    selected = model_picker(bundle, "evaluation_model")
    metrics = metric_rows(report)
    if "model" in metrics:
        st.dataframe(metrics[metrics["model"] == selected], hide_index=True, width="stretch")
    curves = report.get("curves", {}).get(selected, {})
    first, second = st.columns(2)
    with first:
        st.subheader("ROC curve")
        roc = curves.get("roc", {})
        if roc:
            st.line_chart(pd.DataFrame({"False positive rate": roc["fpr"], "True positive rate": roc["tpr"]}), x="False positive rate", y="True positive rate")
    with second:
        st.subheader("Precision–recall curve")
        pr = curves.get("pr", {})
        if pr:
            st.line_chart(pd.DataFrame({"Recall": pr["recall"], "Precision": pr["precision"]}), x="Recall", y="Precision")
    st.subheader("Probability calibration")
    calibration = curves.get("calibration", {})
    if calibration:
        calibration_frame = pd.DataFrame({"Mean predicted probability": calibration["predicted"], "Observed non-attendance rate": calibration["observed"]})
        st.line_chart(calibration_frame, x="Mean predicted probability", y="Observed non-attendance rate")
    st.caption("Well-calibrated predictions lie near equal predicted and observed rates. A smooth-looking curve alone does not demonstrate reliability.")
    calibrator = manifest.get("calibrators", {}).get(selected)
    if calibrator and calibrator.get("coef", 0) < 0:
        st.warning("This model's sigmoid calibration slope is negative, so calibration reverses its raw score ordering. Calibration was fitted before the final test on the separate calibration period; the inversion and weak final-test performance are disclosed, not corrected by refitting to the test set.")
    if "model" in metrics and "confusion_matrix" in metrics:
        match = metrics.loc[metrics["model"] == selected]
        if not match.empty:
            st.subheader("Confusion matrix at the documented threshold")
            st.dataframe(pd.DataFrame(match.iloc[0]["confusion_matrix"], index=["Actual attended", "Actual missed"], columns=["Predicted attended", "Predicted missed"]))
    capacity_results = report.get("capacity", {}).get(selected, [])
    if capacity_results:
        st.subheader("Performance at limited outreach capacity")
        st.dataframe(pd.DataFrame(capacity_results), hide_index=True, width="stretch")
        st.caption("Precision@K: fraction of the K selected appointments subsequently missed. Recall@K: fraction of all missed appointments captured. This is retrospective targeting performance, not prevented missed visits.")
        st.caption("These research results select up to K appointments separately on each appointment day and use seeded-random tie handling. The interactive planner instead applies a total capacity to the current uploaded batch and resolves equal scores by appointment reference. These are different selection policies; the daily research results do not directly validate the batch planner.")
    st.subheader("Data split, model selection and source")
    for field in ["source", "counts", "splits", "selection", "uncertainty", "subgroups", "feature_importance", "limitations"]:
        if field in report:
            with st.expander(field.replace("_", " ").title()):
                st.write(report[field])
    with st.expander("Full machine-readable report"):
        st.json(report, expanded=False)
    st.download_button("Download full evaluation report", json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False).encode(), "careaccess_evaluation.json", "application/json")
    st.subheader("What this evaluation does not establish")
    st.write("Retrospective prediction does not show that outreach prevents missed appointments. Performance may change across clinics, specialties, time periods or patient groups. Local validation and an appropriately governed prospective evaluation are needed before operational use.")
    st.write("Group comparisons are descriptive and may be unstable in small samples. Unavailable patient linkage can limit control of repeated appointments. Review the source documentation and model cards before drawing conclusions.")

else:
    st.header("About CareAccess ML")
    st.write("An appointment prediction and outreach-planning workbench using Python, scikit-learn, PyTorch and Streamlit. Saved trained models run directly in the application; no generative-AI API supplies the probabilities.")
    st.link_button("View GitHub repository", "https://github.com/ibrahimbaabdullah-jpg/CareAccess-ML")
    st.link_button("View original dataset", "https://data.mendeley.com/datasets/wm6w2fvkfj/1")
    source_info = report.get("source", {})
    st.write(f"Training source: {source_info.get('title', 'See model report')} — {source_info.get('author', '')}. License: {source_info.get('license', 'See source')}. DOI: {source_info.get('doi', '')}.")
    st.caption(f"Model bundle created: {manifest.get('created_utc', 'See manifest')}")
    st.subheader("Data, model release and provenance")
    st.json(manifest, expanded=False)
    st.subheader("Input contract")
    st.json(contract, expanded=False)
    st.subheader("Privacy and safe use")
    st.write("Only enter non-identifying information that you are authorized to process on this hosting service. Do not submit names, contact details, medical record numbers or confidential patient data. Input is held in the active session for prediction and download; it is not added to model training or deliberately written to a database. Hosting providers may retain ordinary technical access logs. Removing names alone does not establish anonymization or institutional approval.")
    st.write("This is not a clinically approved tool. It must not be used to deny care, cancel appointments, punish patients, or make autonomous clinical decisions. A score does not establish that a person will miss a visit.")
    st.write("Training data source, license, cleaning decisions, model cards, environment versions and reproducibility instructions are documented in the repository. Third-party data remains subject to its source license.")

st.divider()
st.caption("CareAccess ML · Historical-data evaluation and saved-model inference · Human review required")

"""Model comparison and evaluation pages."""

import json

import pandas as pd
import streamlit as st

from careaccess.ui.common import (
    label,
    metric_rows,
    model_picker,
)


def render_comparison(bundle):
    report = bundle["report"]
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


def render_evaluation(bundle):
    report = bundle["report"]
    manifest = bundle["manifest"]
    st.header("Inspect the evidence and limitations")
    st.write("These measured results come from held-out historical Mendeley appointments. Interactive inputs and uploaded files do not change these evaluation results.")
    selected = model_picker(bundle, "evaluation_model", scoring=False)
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
        st.warning("This model's sigmoid calibration slope is negative, so calibration reverses its raw score ordering. Calibration was fitted before the final test on the separate calibration period; the inversion and weak final-test performance are disclosed, not corrected by refitting to the test set. Because its calibrated scores rank appointments in reverse, it is not offered for Live or Batch Prediction.")
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

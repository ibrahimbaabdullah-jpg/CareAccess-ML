"""Home dashboard: purpose, headline evidence and limitations."""

import streamlit as st

from careaccess.planner import export_csv
from careaccess.ui.common import (
    RESEARCH_DEMO_URL,
    empty_template,
    label,
    metric_rows,
)


def render_home(bundle):
    report = bundle["report"]
    st.header("Plan supportive outreach, one appointment at a time")
    st.write(
        "Explore real trained models that estimate missed-appointment probability, then build a review list that fits your team's reminder-call capacity."
    )
    st.info(
        "This app presents real trained models and aggregate results from publicly released appointment research datasets. The deployed models use the original Mendeley source; Vitória is a separate aggregate benchmark. Underlying patient records and generated appointment cohorts are not published here. Prediction performance does not prove that outreach prevents missed visits."
    )
    st.caption(
        "Input warnings concern data submitted to this host. They do not classify all displayed research aggregates or model outputs as sensitive."
    )
    st.subheader("Three simple steps")
    cols = st.columns(3)
    for col, title, body in zip(
        cols,
        ["1 · Enter an appointment", "2 · Validate a schedule", "3 · Choose your outreach capacity"],
        [
            "Open Live Prediction to run a saved trained model on permitted inputs.",
            "Use Batch Prediction to check a permitted nonsensitive, non-identifying CSV.",
            "Open Outreach Planner to create and download a ranked review list.",
        ],
        strict=True,
    ):
        col.markdown(f"**{title}**")
        col.write(body)
    counts = report.get("counts", {})
    counts_cols = st.columns(3)
    counts_cols[0].metric(
        "Source appointment records", f"{counts['original_rows']:,}" if "original_rows" in counts else "Not available"
    )
    counts_cols[1].metric(
        "Records after documented cleaning", f"{counts['clean_rows']:,}" if "clean_rows" in counts else "Not available"
    )
    counts_cols[2].metric("Default trained model", label(bundle["default_model"]))
    st.caption("These are appointment records, not a count of unique patients.")
    st.subheader("Original deployed-model performance")
    rows = metric_rows(report)
    if not rows.empty:
        st.dataframe(rows, hide_index=True, width="stretch")
    else:
        st.warning("Evaluation metrics are not available in this model bundle.")
    st.write(f"Default model: **{label(bundle['default_model'])}**")
    st.caption(
        "See Model Comparison and Model Evaluation for test design, thresholds, probability reliability and limitations."
    )
    st.download_button(
        "Download empty CSV template", export_csv(empty_template()), "careaccess_empty_template.csv", "text/csv"
    )
    st.subheader("Separate research: Vitória historical benchmark")
    st.write(
        "A frozen, eight-date benchmark found a ranking signal against a simple rule, but did not establish extra value over a fixed logistic comparator. It does not replace or validate the original models above."
    )
    st.caption(
        "Open Research / Vitória in the sidebar for aggregate findings, uncertainty, probability overprediction and allocation limits."
    )
    st.link_button("Open separate browser-only research demo", RESEARCH_DEMO_URL)
    st.caption(
        "Manually explore the frozen Vitória XGBoost in your browser. It uses a different cohort and does not replace or validate the original Mendeley models above."
    )

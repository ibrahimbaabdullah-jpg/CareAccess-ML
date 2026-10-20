"""Read-only display of a bounded, checksummed aggregate research summary.

No research weights, source rows or inference path are loaded by this module.
"""

import json
from hashlib import sha256
from pathlib import Path

import pandas as pd
import streamlit as st

SUMMARY_PATH = Path(__file__).resolve().parents[1] / "reports/research/vitoria/summary.json"
SUMMARY_SHA256 = "316d10de679e08019458befb07464b91c18139155db7120fa36e382f99f041dc"


def load_research_summary(path=None):
    """Fail closed if the fixed public aggregate evidence is absent or altered."""
    target = SUMMARY_PATH if path is None else Path(path)
    with target.open("rb") as handle:
        payload = handle.read(32_769)
    if len(payload) > 32_768 or sha256(payload).hexdigest() != SUMMARY_SHA256:
        raise ValueError("Research summary integrity check failed.")
    return json.loads(payload)


def policy_table(summary, identifiers=None):
    """Presentation only: derive percentages from the frozen aggregate facts."""
    rows = []
    for policy in summary["policies"]:
        if identifiers is not None and policy["id"] not in identifiers:
            continue
        captured = policy["captured_events"]
        rows.append(
            {
                "Policy": policy["label"],
                "Recorded no-shows identified": f"{captured:,.2f} expected"
                if policy["capture_is_expectation"]
                else f"{captured:,}",
                "Precision in selected slots": f"{policy['precision']:.2%}",
                "Share of all recorded no-shows": f"{policy['capture_fraction']:.2%}",
            }
        )
    return pd.DataFrame(rows)


def render_research():
    st.header("Vitória research benchmark")
    st.caption("SOURCE-SPECIFIC · HISTORICAL · AGGREGATE EVIDENCE")
    st.info(
        "This separate benchmark does not power Live Prediction, Batch Prediction or Outreach Planner. Those pages still use the original Mendeley-trained models and their original input contract. The two cohorts and their results are not pooled or directly comparable."
    )
    st.link_button("Open separate browser-only research demo", "https://ibrahimbaabdullah-jpg.github.io/CareAccess-ML/")
    st.caption(
        "A separate manual interface for the frozen Vitória XGBoost, evaluated in your browser. Its distinct cohort does not replace or validate the original Mendeley models. This link does not load its model assets into Streamlit."
    )
    try:
        summary = load_research_summary()
    except (OSError, ValueError):
        st.error(
            "The verified research summary is unavailable. Research results cannot be displayed. Original saved-model prediction remains unchanged."
        )
        return

    cohort = summary["cohort"]
    primary = next(row for row in summary["policies"] if row["id"] == summary["selected_id"])
    probability = summary["primary_probability_metrics"]
    st.subheader("A ranking signal, with important limits")
    st.write(
        "The frozen primary XGBoost model identified more recorded no-shows than the locked simple rule at the same research capacity. It did not establish added value over the fixed logistic comparator. No model was reselected after seeing these final outcomes."
    )
    first, second, third = st.columns(3)
    first.metric("Recorded no-shows identified", f"{primary['captured_events']:,} / {cohort['slots']:,}")
    first.caption(
        f"{primary['precision']:.2%} of selected appointments; {primary['capture_fraction']:.2%} of all recorded no-shows"
    )
    second.metric("Gate C · historical ranking", summary["gates"]["C"])
    second.caption("Prespecified screen against random and the locked simple rule")
    third.metric("Gate D · added complexity", summary["gates"]["D"])
    third.caption("Boosting did not establish extra value over fixed logistic regression")
    st.warning(
        "Research only. These are no-shows identified retrospectively, not missed visits prevented. Clinical effectiveness, fairness and operational readiness were not established. Keep ordinary reminders and access to care available to everyone."
    )

    st.subheader("One frozen cohort, equal capacity")
    st.write(
        f"{cohort['appointments']:,} eligible adult appointment records · {cohort['events']:,} recorded no-shows ({cohort['prevalence']:.2%}) · {cohort['observed_dates']} appointment dates across {cohort['observed_iso_weeks']} ISO weeks, 30 May–8 June 2016."
    )
    st.caption(
        "12,555 supplied patient keys, not verified unique people. Some supplied keys occur on multiple appointments."
    )
    st.write(
        "Capacity was the floor of 10% of eligible appointments on each date, totaling 1,498 slots. The same fixed hash-based tie policy was used for each research method. This differs from the live planner's total batch capacity and appointment-reference tie order."
    )
    st.dataframe(
        policy_table(summary, {"xgb_d3", "location_lead_prior", "lr_c1", "matched_random"}),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Precision: recorded no-shows among selected appointments. Capture: selected no-shows divided by all 3,906 recorded no-shows. Random is the exact same-date expectation, so its count is fractional; it is not a sampled favorable list or a deployable probability model."
    )

    st.subheader("How large is the difference?")
    for comparison in summary["comparisons"][:2]:
        comparator = next(row for row in summary["policies"] if row["id"] == comparison["comparator"])
        date = comparison["date_ci95"]
        crossed = comparison["crossed_patient_date_ci95"]
        st.markdown(f"**Primary XGBoost versus {comparator['label'].lower()}**")
        st.write(
            f"{comparison['capture_difference']:+.0f} recorded no-shows identified; {comparison['precision_difference'] * 100:+.2f} percentage points in precision. Paired-date 95% interval: {date['lower'] * 100:+.2f} to {date['upper'] * 100:+.2f} points. Patient/date dependence sensitivity: {crossed['lower'] * 100:+.2f} to {crossed['upper'] * 100:+.2f} points."
        )
    st.caption(
        "The primary beat the locked rule in both weeks, but one of the eight appointment dates favored the rule. The logistic comparison interval includes zero. A failed complexity gate does not establish that logistic regression is superior, and its higher point estimate is not permission to switch the frozen primary model."
    )
    with st.expander("Uncertainty and what the intervals leave out"):
        st.write(summary["uncertainty"]["method"])
        st.write(summary["uncertainty"]["limits"])
        st.write(
            "The prespecified final intervals use 10,000 paired-date replicates and 10,000 crossed patient/date replicates; AUC and average precision use 2,000 date replicates. There was no week bootstrap with only two weeks."
        )
        st.caption(summary["verification"]["scope"])

    st.subheader("Probability reliability and allocation need work")
    columns = st.columns(3)
    columns[0].metric("Primary ROC-AUC", f"{probability['roc_auc']:.3f}")
    columns[1].metric("Primary average precision", f"{probability['average_precision']:.3f}")
    columns[1].caption(f"Recorded no-show prevalence: {cohort['prevalence']:.3f}")
    columns[2].metric("Observed / predicted events", f"{probability['observed_expected_ratio']:.3f}")
    st.write(
        f"Raw probabilities overpredicted: their sum was {probability['expected_events']:,.1f} expected recorded no-shows, compared with {probability['observed_events']:,} observed. Brier score was {probability['brier']:.4f}. No recalibration was applied after the final test."
    )
    st.warning(summary["allocation_warning"])
    st.write(
        f"Temporal shift also matters: appointment month was an unseen category for {summary['design']['unknown_month_count'] / cohort['appointments']:.2%} of final appointments. Unknown-category handling was frozen before final evaluation; it was not repaired using final outcomes."
    )

    with st.expander("All prespecified final policies, including sensitivities"):
        st.dataframe(policy_table(summary), hide_index=True, width="stretch")
        st.caption(
            "The two history variants are sensitivities only. They rely on assumed outcome availability, cannot establish practical benefit, and were not eligible to replace the primary after final evaluation. Constant policies share the same deterministic tie ranking."
        )
    with st.expander("Cohort, timing and the frozen method"):
        st.write(summary["design"]["eligibility"])
        st.write(summary["design"]["timing"])
        st.write(summary["design"]["selection"])
        st.write(
            "30,888 development appointments; ten registered primary candidates; exactly 40 successful learned fits including prespecified history sensitivities and final refits. Development ended 25 May 2016, before the final interval. Procedural target separation was used, not independent cryptographic escrow or a blinded prospective trial."
        )
        st.write(summary["design"]["features"])
        st.write(
            "The source's recorded No-show label is used as supplied. Cancellation, rescheduling and outcome-finalization rules are undocumented; the result is not a fully adjudicated clinical outcome."
        )
        st.caption("Protocol: " + summary["protocol_id"])

    st.subheader("Source, privacy and reuse")
    st.caption(
        "The following source notice and downloadable summary describe this aggregate-only, main-branch Streamlit release. Separate branches or browser-demo releases have their own asset-specific notices and licenses."
    )
    st.write(summary["source"]["notice"])
    st.link_button("View Kaggle source and version history", summary["source"]["url"])
    st.link_button("Read source-data license · CC BY-NC-SA 4.0", summary["source"]["license_url"])
    st.caption(summary["source"]["privacy_notice"])
    st.caption(
        "The repository's MIT license covers its original software; it does not relicense the source data or separately distributed dataset-derived assets. This Streamlit research page distributes only a bounded, cohort-wide aggregate summary, with no Vitória source records or research weights."
    )
    st.download_button(
        "Download aggregate research summary",
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8"),
        "careaccess_vitoria_aggregate_summary.json",
        "application/json",
    )
    with st.expander("ملخص بالعربية"):
        st.write(
            "هذه تجربة بحثية تاريخية منفصلة على بيانات فيتوريا، ولا تشغّل التنبؤات الحالية في الموقع. النموذج المثبّت التقط 544 حالة عدم حضور من 1,498 موعدًا مختارًا، مقابل 460 للقاعدة البسيطة و567 للمقارن اللوجستي. اجتاز اختبار الترتيب التاريخي، لكنه لم يثبت فائدة التعقيد الإضافي. لم نغيّر النموذج بعد رؤية النتيجة. فترة الاختبار قصيرة، والاحتمالات مبالغ فيها، والتوزيع العمري يحتاج إلى مراجعة. لم تثبت فائدة سريرية أو عدالة أو جاهزية للتشغيل الفعلي، وتبقى نتيجة النموذج الأصلي قريبة من الصدفة."
        )

# CareAccess ML
## Missed-Appointment Prediction and Outreach Planner

A functional Python/Streamlit application using **real saved scikit-learn and PyTorch models**, trained on the original Mendeley appointment dataset. No generated patient examples are distributed. The website presents actual aggregate evaluation evidence and accepts authorized nonidentifying inputs.

**Important result:** the frozen final-test models discriminate at approximately chance. The selected logistic model has ROC-AUC 0.505 and average precision 0.169 versus 0.161 test prevalence. Working software is not evidence of useful operational targeting. This release is a transparent, reproducible evaluation workbench; it is not validated for real-patient deployment or effective outreach prioritization.

## Status
[Open the public application](https://careaccess-ml.streamlit.app/). This repair revision has **145 passing local tests** using private actual-source adult inputs; public/CI mode passes **95 tests and explicitly skips 50 source-dependent checks**. Local displayed values and exact CSV download-button payloads agree across all four models. Those local checks do not establish final repaired-deployment acceptance. The earlier hosted baseline passed inference/page/reboot checks; subsequent supported browser downloads verified actual batch CSV bytes for all three learned models within `1e-6`. Earlier blocked attempts and later successful baseline checks are both documented. Verify the deployment's exact repair commit before treating the fixes as live. See [release acceptance status](docs/release-status.md) and the [30-item acceptance matrix](docs/acceptance-matrix.md).

The [bounded model-improvement investigation](docs/predictive-value-investigation.md) did not establish a reliable benefit or qualify a replacement. The original trained artifacts and historical evaluation are preserved. [Independent validation and next-evidence requirements](docs/validation-protocol.md) explain why a genuinely new cohort is needed. [Reachable-history and security review](docs/security-history-audit.md) records repaired risks and remaining provider/dependency/approval gaps. No paid APIs or generative AI services are required.

## Features
- Home dashboard with actual benchmark evidence and clear limitations
- Single appointment prediction through saved models
- CSV schema checks and explicit errors; the entire file must pass validation before scoring
- Capacity-limited supportive outreach review list and CSV export
- Comparison of constant baseline, logistic regression, gradient boosting and PyTorch neural network
- ROC/PR/calibration curves, confusion matrices, date-cluster intervals, subgroup and daily top-K evaluation
- Data source, model cards, reproducibility and privacy documentation

## Quick start (Python 3.12)
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip==26.2.1
pip install -r requirements-dev.txt
python -m pytest -q
python scripts/verify_release.py
streamlit run app.py
```
Dependencies use the official PyTorch CPU wheel index. No GPU required. Tested package versions are pinned. `requirements-lock.txt` records the complete local environment; deployment uses requirements.txt.
PyArrow is explicitly pinned to 24.0.0, matching the hosting provider's reported
compatibility workaround; the local application and real-source inference suite
were rerun with this version. Development/build tooling uses setuptools 83.0.0.

For reproducible hosting setup, release acceptance and rollback, see the
[deployment guide](docs/deployment.md).

## Data and training
Salazar, Luiz Henrique (2023), *Medical Appointments No-Show*, Mendeley Data, V1, [DOI 10.17632/wm6w2fvkfj.1](https://data.mendeley.com/datasets/wm6w2fvkfj/1), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

49,593 original appointment records, 26 columns; 2,921 excess exact duplicate rows removed, leaving 46,672. They are appointments, not necessarily unique patients. Original data and row-level transformed records are deliberately absent from this repository. The dataset contains sensitive quasi-identifiers; its public availability is not permission to redistribute patient-level information.

See [data audit](docs/data-audit.md), [retraining guide](docs/retraining.md), [full report](reports/evaluation-report.md), [machine-readable results](reports/evaluation.json), and the four model cards in docs/.

The raw age field was inconsistent with historical appointment dates. Age was recomputed locally from birth date, then only numeric age used; birth dates are never published. Inputs are scheduled specialty, age and calendar/time features. Outcomes, reasons for nonattendance and observed weather are excluded. A follow-up development-only disability/city ablation did not improve the basic model; see reports/feature_ablation.json.

## CSV format
Use the header-only `examples/appointments_template.csv`. No patient rows are provided. Exact columns:
`appointment_id,appointment_datetime,specialty,age`.
References must be temporary nonidentifying tokens containing letters, digits, hyphens or underscores (up to 40 characters). Date/time uses local `YYYY-MM-DDTHH:MM`; age 0–110 or blank; specialty is a supported source category or blank. No names, birth dates, outcome labels, health identifiers or additional columns. Use only information you are authorized to process on the hosting service, without identifiers or identifiable health information. Do not upload sensitive childhood records.
CSV files must be UTF-8, with one appointment per physical line, exactly four
unique template columns and no cell longer than 128 characters. The parser
checks these bounds before constructing a table. Missing age/specialty and dates
outside the training period are flagged beside each scored appointment.

Planner capacity is a total across the uploaded batch, including all dates;
equal scores use appointment-reference order. Review statuses persist while
navigating or changing capacity and reset for a new scoring attempt. The daily
top-K evaluation is a separate research policy with seeded-random tie handling,
so it is not direct validation of the batch planner's selection policy.

## Privacy and safety
Uploads processed in session memory, not deliberately stored or logged. Streamlit itself processes uploaded bytes; do not treat a public hosting service as an approved clinical environment. Model artifacts are fixed, checksummed and loaded with skops/safetensors; arbitrary model uploads are not accepted. CSV formula injection escaped. No automatic calls/SMS, cancellations, penalties or eligibility decisions. Population differences and dataset shift can invalidate probabilities.

## Arabic summary
CareAccess ML تطبيق لتجربة توقع عدم حضور المواعيد باستخدام نماذج مدربة فعليًا، مع تنظيم قائمة تواصل داعمة. نتائج التمييز الحالية قريبة من الصدفة؛ لذلك لا يصح تسويقه كأداة أثبتت فعاليتها في ترتيب المرضى. الموقع يعرض التقييم الحقيقي ويستقبل مدخلات غير مُعرِّفة ومصرّحًا بمعالجتها، ويعرض النتائج الحقيقية والقيود بوضوح، ولا يلغي المواعيد أو يتخذ قرارات علاجية.

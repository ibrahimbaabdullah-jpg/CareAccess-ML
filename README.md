# CareAccess ML
## Missed-Appointment Prediction and Outreach Planner

A functional Python/Streamlit application using **real saved scikit-learn and PyTorch models**, trained on the original Mendeley appointment dataset. No generated patient examples are distributed. The website presents actual aggregate evaluation evidence and accepts only permitted nonsensitive, non-identifying inputs on the public host.

**Important result:** the frozen final-test models discriminate at approximately chance. The selected logistic model has ROC-AUC 0.505 and average precision 0.169 versus 0.161 test prevalence. Working software is not evidence of useful operational targeting. This release is a transparent, reproducible evaluation workbench; it is not validated for real-patient deployment or effective outreach prioritization.

## Status
[Open the public application](https://careaccess-ml.streamlit.app/). Security and acceptance repairs have been merged; exact-main CI and protected-branch/read-only-workflow controls were verified. Actual hosted batch CSV bytes for all three learned models and planner export match saved-model inference within `1e-6`.

The previous session-state revision preserved chosen capacity and explicit consent across page navigation, while Clear/revoke still removed batch state. That revision passed **156 local tests** with private actual-source adult inputs; public mode passed **101 and explicitly skipped 55 source-dependent checks**. Its exact protected-main CI and focused hosted recheck are recorded in the existing release evidence. See [release status](docs/release-status.md), [session-state verification](reports/acceptance/capacity-consent-verification.json), and the [30-item acceptance matrix](docs/acceptance-matrix.md).

The [bounded model-improvement investigation](docs/predictive-value-investigation.md) did not qualify a replacement or establish reliable prioritization benefit. Original trained models and historical evaluation are preserved. [Independent review](docs/validation-protocol.md) explains why new evidence is needed; [security/history review](docs/security-history-audit.md) distinguishes verified controls from provider/dependency limits. No paid API or generative AI service is required.

## Separate research evidence: Vitória
The **Research / Vitória** page reports a frozen source-specific historical benchmark on a different cohort: 15,016 eligible adult appointments over eight dates in two weeks. At 1,498 fixed research slots, the frozen primary XGBoost identified 544 recorded no-shows, the locked simple rule 460, and fixed new-source logistic regression 567. The historical ranking gate passed; the added-complexity gate failed. There was no post-test reselection or change to deployed inference.

This is aggregate research evidence, not a before/after improvement of the original 0.505-AUC model, an external validation of its weights, or proof of prevented no-shows. Raw probabilities overpredicted, allocation concentrated in younger adults, and sparse temporal coverage limits the inference. See the [benchmark report and source/license notice](docs/vitoria-benchmark.md) and [bounded aggregate summary](reports/research/vitoria/summary.json). No Vitória source records, individual scores or weights are distributed. The source is credited to Joni Hoppen and Aquarela Analytics under CC BY-NC-SA 4.0; independent software remains MIT.

The aggregate research integration passes **176 local tests** using the existing private offline source fixture. Public mode passes **120 tests and explicitly skips 56 source-dependent checks**. Its release-integrity check and dependency consistency check pass. These are local results; exact-commit CI and hosted visual checks must be recorded separately after publication. No new patient records were sent to the public host for this update.

## Features
- Home dashboard with actual benchmark evidence and clear limitations
- Single appointment prediction through saved models
- CSV schema checks and explicit errors; the entire file must pass validation before scoring
- Capacity-limited supportive outreach review list and CSV export
- Comparison of constant baseline, logistic regression, gradient boosting and PyTorch neural network
- ROC/PR/calibration curves, confusion matrices, date-cluster intervals, subgroup and daily top-K evaluation
- Separate, read-only Vitória research benchmark with frozen aggregate findings and uncertainty
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
References must be temporary nonidentifying tokens containing letters, digits, hyphens or underscores (up to 40 characters). Date/time uses local `YYYY-MM-DDTHH:MM`; age 0–110 or blank; specialty is a supported source category or blank. No names, birth dates, outcome labels, health identifiers or additional columns. Do not upload patient records or enter patient-level health information on the public host, even when publicly available or stripped of names. Use only permitted nonsensitive, non-identifying inputs.
CSV files must be UTF-8, with one appointment per physical line, exactly four
unique template columns and no cell longer than 128 characters. The parser
checks these bounds before constructing a table. Missing age/specialty and dates
outside the training period are flagged beside each scored appointment.

Planner capacity is a total across the uploaded batch, including all dates;
equal scores use appointment-reference order. Review statuses persist while
navigating or changing capacity and reset for a new scoring attempt. The daily
top-K evaluation is a separate research policy with seeded-random tie handling,
so it is not direct validation of the batch planner's selection policy.

## Public-host input boundary
This public Streamlit demo must not receive or process sensitive personal information, including patient-level health information. Do not upload patient records or enter patient health details. Public availability, removing names, or checking a consent box does not establish that data is suitable for this host. Source-data evaluation is performed offline. Real-patient use requires a separately governed deployment and review.

[Streamlit Community Cloud Addendum 4](https://streamlit.io/terms-of-use) restricts sensitive personal information, including health information. A consent checkbox does not override provider terms. A controlled local/institutional deployment still needs appropriate privacy, security, legal and clinical review; local execution alone is not a compliance claim. No new source-derived inputs are used for hosted QA. Historical hosted checks are retained as historical evidence, not permission to repeat those uploads.

## Privacy and safety
Uploads processed in session memory, not deliberately stored or logged. Streamlit itself processes uploaded bytes; do not treat a public hosting service as an approved clinical environment. Model artifacts are fixed, checksummed and loaded with skops/safetensors; arbitrary model uploads are not accepted. CSV formula injection escaped. No automatic calls/SMS, cancellations, penalties or eligibility decisions. Population differences and dataset shift can invalidate probabilities.

## Arabic summary
CareAccess ML تطبيق لتجربة توقع عدم حضور المواعيد باستخدام نماذج مدربة فعليًا، مع تنظيم قائمة تواصل داعمة. نتائج التمييز الحالية قريبة من الصدفة؛ لذلك لا يصح تسويقه كأداة أثبتت فعاليتها في ترتيب المرضى. الموقع يعرض التقييم الحقيقي ويستقبل مدخلات غير مُعرِّفة ومصرّحًا بمعالجتها، ويعرض النتائج الحقيقية والقيود بوضوح، ولا يلغي المواعيد أو يتخذ قرارات علاجية.

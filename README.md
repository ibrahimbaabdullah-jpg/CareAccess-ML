# CareAccess ML

A research workbench for appointment no-show risk and capacity-limited reminder outreach. It trains and compares real models, reports their measured performance honestly, and offers an interactive prediction and outreach-planning demo.

مشروع بحثي يقدّر احتمال عدم حضور المواعيد ويدرس اختيار المواعيد ذات الأولوية للتذكير عند محدودية الموارد. يقارن نماذج مدرّبة فعلًا، ويعرض أداءها وحدودها بشفافية، مع تجربة تفاعلية للتنبؤ وتخطيط التواصل.

- **[Streamlit app](https://careaccess-ml.streamlit.app/)**: the original Mendeley-trained models and outreach planner.
- **[Browser-only research demo](https://ibrahimbaabdullah-jpg.github.io/CareAccess-ML/)**: a frozen XGBoost trained on the separate Vitória cohort (served from the [`research-browser-demo` branch](https://github.com/ibrahimbaabdullah-jpg/CareAccess-ML/tree/research-browser-demo)).

## Key results / أهم النتائج

| Cohort | What was measured | Result |
|---|---|---|
| Mendeley (deployed app) | Frozen final test, 8,905 appointments from 2021–2022 | Logistic regression ROC-AUC **0.505**, average precision 0.169 vs 0.161 prevalence: **close to chance** |
| Vitória (research only) | 15,016 adult appointments, 1,498 fixed outreach slots | XGBoost identified **544** recorded no-shows vs **460** for a simple rule and **567** for logistic regression |

Working software is not evidence of useful targeting. Clinical effectiveness and reductions in missed appointments have **not** been demonstrated. The deployed Mendeley models have only five inputs (age, hour, weekday, month, specialty) and no booking lead time or attendance history, which likely explains much of their weak signal. The Gradient Boosting model's calibration slope is negative (it reverses rankings), so it appears only on the evaluation pages and cannot be selected for scoring.

نتائج النموذج الحالي قريبة من الصدفة، ولم تُثبت الفعالية السريرية أو القدرة على خفض المواعيد الفائتة. لا يصح استخدامه كأداة لترتيب المرضى.

## Quick start (Python 3.12)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
ruff check . && ruff format --check .
python -m pytest -q
python scripts/verify_release.py
streamlit run app.py
```

Dependencies are pinned and use the PyTorch CPU wheel index; no GPU is needed. `requirements.txt` is used for deployment and `requirements-lock.txt` records the full environment. Source-dependent tests skip unless `CAREACCESS_TEST_DATA` points at the private fixture. See the [deployment guide](docs/deployment.md) and [retraining guide](docs/retraining.md).

## Project layout

| Path | Contents |
|---|---|
| `app.py` | Streamlit interface |
| `careaccess/` | Input validation, CSV parsing, inference, evaluation, planner, session state, research page |
| `scripts/train.py` | Reproducible chronological training and evaluation |
| `models/` | Checksummed artifacts (`skops`, `safetensors`) and `manifest.json` |
| `reports/` | Evaluation results, research summaries and audit evidence |
| `docs/` | Model cards, data audit, validation protocol, privacy and release notes |

## Features

- Single-appointment and batch CSV prediction with saved models
- Strict CSV schema checks; the whole file must pass before scoring
- Capacity-limited outreach review list with CSV export
- Comparison of a constant baseline, logistic regression, gradient boosting and a PyTorch neural network
- ROC/PR/calibration curves, confusion matrices, date-cluster intervals, subgroup and daily top-K evaluation
- Read-only Vitória research benchmark with frozen aggregate findings and uncertainty

## Data

Salazar, Luiz Henrique (2023), *Medical Appointments No-Show*, Mendeley Data, V1, [DOI 10.17632/wm6w2fvkfj.1](https://data.mendeley.com/datasets/wm6w2fvkfj/1), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). 49,593 records; 2,921 exact duplicates removed, leaving 46,672 appointments (not unique patients). Age is recomputed from the date of birth and only numeric age is used; birth dates are never published. Original and row-level transformed records are not in this repository. Splits are chronological: train ≤2018, validation 2019, calibration 2020, test 2021–2022.

The Vitória source is credited to Joni Hoppen and Aquarela Analytics under CC BY-NC-SA 4.0. This repository's MIT license covers the original software only; it does not relicense source data or dataset-derived assets.

Details: [data audit](docs/data-audit.md), [evaluation report](reports/evaluation-report.md), [model-improvement investigation](docs/predictive-value-investigation.md), [Vitória benchmark](docs/vitoria-benchmark.md), [validation protocol](docs/validation-protocol.md) and the model cards in `docs/`.

## CSV format

Start from the header-only `examples/appointments_template.csv`. Columns: `appointment_id,appointment_datetime,specialty,age`.

- `appointment_id`: a temporary, non-identifying token (letters, digits, `-`, `_`; up to 40 characters)
- `appointment_datetime`: local time as `YYYY-MM-DDTHH:MM`
- `age`: 0–110, or blank
- `specialty`: a supported source category, or blank

Files must be UTF-8, one appointment per line, at most 5 MB / 5,000 rows, with no cell longer than 128 characters. Missing fields and dates outside the training period are flagged beside each score.

Planner capacity is a total across the uploaded batch, including all dates, and equal scores are ordered by appointment reference. The daily top-K evaluation is a separate research policy with seeded-random tie handling, so it does not directly validate the planner's selection policy.

## Privacy and safe use

The public Streamlit demo must not receive sensitive personal information, including patient-level health information. Do not upload patient records or enter patient health details. Public availability, removing names, or checking a consent box does not establish that data is suitable for this host. Source-data evaluation is performed offline. [Streamlit Community Cloud Addendum 4](https://streamlit.io/terms-of-use) restricts sensitive personal information, and a consent checkbox does not override provider terms. Real-patient use requires a separately governed deployment with privacy, security, legal and clinical review.

Uploads are processed in session memory and are not deliberately stored or logged. Model artifacts are checksummed and loaded with an allowlist; arbitrary model uploads are not accepted. CSV exports escape spreadsheet formulas. The app never sends messages, cancels appointments or makes eligibility decisions. See [privacy and security](docs/privacy-security.md) and [release status](docs/release-status.md).

لا تُرفع سجلات مرضى ولا معلومات صحية على الاستضافة العامة. يُسمح فقط بمدخلات غير حساسة وغير مُعرِّفة. التطبيق لا يلغي المواعيد ولا يتخذ قرارات علاجية.

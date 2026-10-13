# Reproduce the benchmark

Use Python 3.12 and install `requirements-dev.txt`. Acquire version 1 from https://data.mendeley.com/datasets/wm6w2fvkfj/1. Review the CC BY 4.0 terms and save the original CSV outside the repository. The source contains sensitive quasi-identifiers and childhood records: handle it locally under appropriate governance, never publish raw records or upload them to the public app.

```bash
python scripts/acquire_data.py --output ../careaccess-private/medical-appointments-no-show-en.csv
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python scripts/train.py --data ../careaccess-private/medical-appointments-no-show-en.csv
python scripts/feature_ablation.py ../careaccess-private/medical-appointments-no-show-en.csv
python scripts/make_report.py
python -m pytest -q
```

Source SHA256: `903f30aa008c478dd6451550b53c2bf937653f28763ea288e43f5e99c536cbdf`. Preparation enforces this checksum. Dates use strict day-first parsing. Deduplication removes exact full-row duplicates only; identical distinct encounters may still be collapsed. There are 46,672 retained appointments, zero removals for invalid target/date, and 8,706 missing derived ages. Gender is retained locally only for aggregate audit. No patient identities or historical attendance linkage are inferred.

## Frozen temporal design
- Training: 2016–2018, 29,187 appointments
- Validation: 2019, 7,753 appointments
- Calibration: 2020, 827 appointments
- Final test: 2021–2022, 8,905 appointments

Preprocessing is fitted on training data only. Dates are never shuffled across splits. Seed: 20261013; CPU threads: two.

## Fixed training settings
- Logistic regression: C=1, maximum 1,000 iterations
- Histogram gradient boosting: 150 iterations, 15 leaves, minimum leaf size 40, L2=2, no internal early stopping
- PyTorch: hidden layers 32/16 with ReLU, dropout 0.1, AdamW learning rate 0.001, weight decay 0.001, batch size 256, maximum 40 epochs; validation log-loss early stopping with patience six; retain best weights
- Constant baseline: training-set missed-appointment prevalence

The default is selected by validation average precision, then log loss/name to break ties. Sigmoid calibration and an F1 threshold grid are fitted on 2020 only. Final-test results never change selection. Calibration is uncertain because 2020 is sparse and may reverse weak raw ranking, as occurred for gradient boosting. Do not retune against the published final test: new development needs a genuinely new evaluation cohort.

Evaluation code is in `careaccess/evaluation.py`. Confidence intervals use 200 date-cluster bootstrap replicates and a fixed seed; no patient-cluster interval is possible. Daily top-K evaluation uses common label-independent seeded tie breaks, with an exact expected-random-selection comparator. Identified missed appointments are not claimed to have been prevented.

## Ordering disclosure
The first automated test report preceded a requested follow-up disability/city sensitivity analysis. That analysis used only 2016–2019 observations for fitting/evaluation and changed no selected model or feature. A verification run repeated the unchanged model fit and corrected only capacity tie handling. This is not a prospectively registered study.

## Artifact and numerical verification
Checksums and configuration hashes are in `models/manifest.json`. Byte hashes may change across serialization runs. A private fixture of actual adult-source rows is used for numerical checks; it is never distributed. Set `CAREACCESS_TEST_DATA` to its local path to run source-dependent tests. Those tests explicitly skip when the fixture is absent. Aggregate parity evidence is in `reports/inference-verification.json`.

No generated patient examples are substituted. Revalidate numerical tolerance (absolute 1e-6), behavior and dependency compatibility after changing library versions or hardware. Public UI text revisions did not retrain the saved models; the manifest preserves original training-configuration hashes alongside the current configuration hashes.

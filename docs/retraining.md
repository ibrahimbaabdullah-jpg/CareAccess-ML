# Reproduce the benchmark

Use Python3.12 and install requirements-dev.txt. Acquire version1 only from https://data.mendeley.com/datasets/wm6w2fvkfj/1, inspect terms/CCBY4.0 and save the original CSV outside the repository. The source contains sensitive and childhood records; handle locally with appropriate governance, never publish raw data or upload it to the public app.

```bash
python scripts/acquire_data.py --output ../careaccess-private/medical-appointments-no-show-en.csv
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python scripts/train.py --data ../careaccess-private/medical-appointments-no-show-en.csv
python scripts/feature_ablation.py ../careaccess-private/medical-appointments-no-show-en.csv
python scripts/make_report.py
python -m pytest -q
```

Source SHA256: 903f30aa008c478dd6451550b53c2bf937653f28763ea288e43f5e99c536cbdf. Source hash enforced by preparation. Strict day-first parsing. Exact full-row dedup only (identical distinct encounters may be collapsed). 46,672 retained; invalid target/date removal0, derived age missing8,706. Gender retained only for local aggregate audit. No patient-linkage features or identities inferred.

Frozen dates: train2016–2018 (29,187), validation2019 (7,753), calibration2020 (827), finaltest2021–2022 (8,905). Preprocessing fitted on training only. No shuffle across dates. Seed20261013, CPU two threads. Logistic C1,maxiter1000; HistGB150iterations,15leaves,minleaf40,L2=2,nointernalearlystop; PyTorch32/16ReLU,dropout0.1,AdamWlr0.001,weightdecay0.001,batch256,max40epochs,validationlogloss patience6, savedbestweights. Constant trainingprevalence baseline.

Select model by uncalibrated validationAP, tie by logloss/name. Fit sigmoid calibration and F1-grid threshold on2020 only. Test evaluated without changing selection. Low-parameter calibration has substantial uncertainty and may reverse weak raw ranking (HistGB here). Do not retune on the published final-test results; new development requires a genuinely new held-out evaluation.

Full aggregation routines in careaccess/evaluation.py. Bootstrap200dateclusters, seedfixed; no patientclusterCI. Same-day topK with label-independent seeded tie breaks and exact expected random selection comparator. No assertion that identified missed appointments were prevented.

The first automated test report preceded a requested follow-up disability/city sensitivity analysis. That analysis used only2016–2019 data for fitting/evaluation and changed no selected model or features. This ordering is explicitly disclosed; it is not a prospectively registered study.

Model checksums and config hashes reside in models/manifest.json. Artifact byte hashes can change across serialization runs; deterministic numerical outputs are compared within absolute1e-6 using a private actual-source fixture, never distributed. Set CAREACCESS_TEST_DATA to its local path; source-dependent tests explicitly skip when unavailable. Aggregate results reside in reports/inference-verification.json. Run the release verification after generating reports and fixtures, and record results. Revalidate on other library versions or hardware.

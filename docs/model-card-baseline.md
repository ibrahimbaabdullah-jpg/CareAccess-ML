# Model card: Constant baseline

Version 0.1.0. Trained 2026-10-13T08:34:28.272703+00:00. Source Mendeley DOI10.17632/wm6w2fvkfj.1, CC BY4.0.

## Intended use
Reproducible retrospective evaluation and authorized nonidentifying input inference of supportive appointment outreach. Not validated for operational deployment. Never use for eligibility, cancellation, punishment, or reduced access to care.

## Training
29,187 appointments before2019; 7,753 validation2019; 827 calibration2020; 8,905 finaltest2021–2022. Predictors: specialty, age at appointment, scheduled hour, weekday and month. No outcome reasons, observed weather, DOB, diagnoses or direct identifiers enter inference.

## Evaluation
ROC-AUC 0.5000; AP 0.1609; Brier 0.1429. Near-chance discrimination; no proven beneficial prioritization. Full metrics/confidence intervals/subgroups and limitations in evaluation.json.

## Safe artifacts
Bundled sklearn objects loaded with skops type checks; neural weights use safetensors. SHA256 checksums in manifest. No user-uploaded models. Models may encode historical associations and are not guaranteed privacy-preserving.

## Limitations
No reliable patient identifier, possible repeated-person overlap, temporal/pandemic shift, modest/sparse calibration sample, single rehabilitation service, no causal outreach evaluation. Missing age imputed from training only. Unknown specialties rejected; blanks explicitly missing.

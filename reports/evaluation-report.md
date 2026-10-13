# CareAccess ML: frozen evaluation report

## Key finding
The software performs real trained-model inference, but discrimination is approximately chance. Useful operational prioritization is NOT established. Do not deploy for clinical operations without local validation and prospective governance.

## Source and sample
Original 49,593 records; 2,921 exact duplicate rows removed; 46,672 retained. 8,905 held-out appointments from 2021–2022, including 1,433 missed visits (16.09%).

## Models (final test)
|Model|ROC-AUC|Average precision|Brier|Precision|Recall|F1|Threshold|
|---|---:|---:|---:|---:|---:|---:|---:|
|Constant baseline|0.5000|0.1609|0.1429|0.1609|1.0000|0.2772|0.0500|
|Logistic Regression|0.5049|0.1687|0.1353|0.1609|0.9979|0.2772|0.1300|
|Gradient Boosting|0.4870|0.1526|0.1357|0.1609|1.0000|0.2772|0.0500|
|PyTorch Neural Network|0.5013|0.1655|0.1354|0.1601|0.9860|0.2755|0.1300|

Average precision baseline on the test set is its missed-visit prevalence. Accuracy is deliberately not promoted. Thresholds maximize F1 on calibration data; their very high recall and poor specificity do not justify contacting everyone. Capacity selection is separately reported per day.

## Selection and calibration
Highest 2019 validation average precision among the three trained candidates; tie break log loss, then name. No final-test results used.
All candidate hyperparameters fixed before training. PyTorch uses actual AdamW weight updates, two hidden layers (32/16), validation loss early stopping. Train-only imputation/scaling/encoding. Calibration is sigmoid fit on 2020, threshold grid 0.05–0.95 also on 2020. HistGB sigmoid slope is negative in this temporal sample: discrimination was weak/reversed during calibration, further evidence of instability; no test-driven correction was made.

## Audit ordering
An initial automated fixed-config run generated test metrics before the additional-input clarification. Subsequent disability/city sensitivity analysis used only training/2019 validation and did not change the original features or selection rule. Final-test results were not used for feature, model, calibration or threshold selection. A later verification run repeated the unchanged fit and corrected only label-independent ties in capacity evaluation.

## Additional-input necessity
Development-only HistGB ablations considered disability and city. Each reduced 2019 validation AP and worsened Brier; the minimal predictors were retained. This does not prove such features are universally useless. ICD was excluded because preappointment recording time was not established and historical missingness is severe.

## Uncertainty
95% percentile intervals use 200 date-cluster bootstrap replicates of fixed final-test predictions; they are conditional on the fitted models and cannot account fully for repeated patients. Full intervals, subgroup counts, daily Precision@K/Recall@K, curves and matrices are in evaluation.json. Small groups (<100 records or <20 events/non-events) suppressed.

## Limitations
- Single historical Brazilian rehabilitation service; not locally validated elsewhere.
- No patient identifiers: repeated persons may span periods; unseen-patient generalization and patient-cluster intervals cannot be established.
- Exact full-row duplicates removed; some identical distinct encounters may have been collapsed.
- Age recomputed from birth date; provided raw age and all age flags excluded. Birth dates are never published.
- Appointment snapshot availability is assumed, not proven by versioned booking records.
- Outcome prevalence and service activity changed over time, including the pandemic.
- Calibration uses only 827 appointments and 117 missed visits from sparse 2020 data.
- Date-cluster confidence intervals do not account fully for repeated patients.
- Predicted missed attendance is not predicted benefit from outreach. No causal or prospective reduction in no-shows is established.
- Public application requires authorized nonidentifying inputs; identifiable health information must not be uploaded.
- Probabilities outside historical support or in a new institution may be unreliable.
- Daily top-K evaluation counts potential missed visits identified, not prevented.

## Reproducibility
See docs/retraining.md, models/manifest.json, reports/source-audit.json and reports/feature_ablation.json. No row-level original/cleaned data or held-out patient predictions are included.

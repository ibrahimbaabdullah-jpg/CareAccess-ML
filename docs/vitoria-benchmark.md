# Vitória: frozen historical ranking benchmark

## Status and separation from the application model

This is a **source-specific, short historical ranking pilot**, reported as cohort-wide aggregate research evidence. It is not external validation of the original CareAccess weights. It is not evidence of clinical benefit, fairness, prevented no-shows, probability deployment or operational readiness.

The original Mendeley-trained models, preprocessing, input contract and historical evaluation remain unchanged. The original default logistic model still has final-test ROC-AUC **0.505**, close to chance, and is not validated for real-patient prioritization. Live Prediction, Batch Prediction and Outreach Planner do not use the Vitória research model. The two sources, populations and evaluation windows differ; their metrics must not be pooled or treated as a before/after improvement.

## Final result

The frozen primary was **xgb_d3**, selected on three expanding, purged temporal development folds. The strongest practical simple rule was locked before final outcomes were opened. The fixed **lr_c1** comparator was trained on the new source; it is not the original deployed logistic artifact.

Final eligibility was recorded adult age 18–110, valid supplied linkage keys and calendar dates, and booking at least three source calendar days before the appointment. The final interval, **30 May–8 June 2016**, contains **15,016 appointments**, **12,555 supplied patient keys**, **3,906 recorded no-shows**, **eight observed appointment dates**, and **two ISO weeks**. Supplied patient keys are not verified unique people.

The research policy allocated floor(10% of eligible appointments) on each date, totaling **1,498 slots**, with fixed hash-based tie breaking. It is different from the interactive planner's global batch capacity and appointment-reference tie policy.

| Prespecified policy | Recorded no-shows identified | Precision in selected slots | Share of all recorded no-shows |
|---|---:|---:|---:|
| Frozen primary XGBoost | 544 | 36.32% | 13.93% |
| Locked location + lead-time rule | 460 | 30.71% | 11.78% |
| Fixed new-source logistic regression | 567 | 37.85% | 14.52% |
| Exact same-date random expectation | 389.67 expected | 26.01% | 9.98% |

The random count is a mathematical expectation, not a sampled list or deployable probability estimate. “Identified” means retrospectively selected appointments carrying the source's recorded no-show label. No calls were made and no prevented no-shows were measured.

- **Gate C: PASS**, the prespecified historical ranking screen against random and the locked practical simple rule. This is a bounded research result, not an adoption decision.
- **Gate D: FAIL**, added value of boosting complexity over the fixed logistic comparator was not established. This does not prove logistic regression is superior. There was no post-test reselection, calibration update or deployment change.

Against the locked rule, primary XGBoost identified **84** additional recorded no-shows, a **+5.61 percentage-point** precision difference. The paired-date 95% interval was **+2.68 to +8.27 points**; crossed patient/date dependence sensitivity was **+0.62 to +10.66 points**. The weekly differences were positive in both weeks, but one of the eight appointment dates favored the rule.

Against fixed logistic regression, XGBoost identified **23 fewer** recorded no-shows, a **−1.54-point** difference. The paired-date 95% interval was **−3.41 to +0.43 points**; crossed dependence sensitivity was **−5.29 to +2.41 points**. These intervals include zero.

The [bounded machine-readable aggregate summary](../reports/research/vitoria/summary.json) includes all ten prespecified final policies, including the two history sensitivities. Their higher point estimates do not authorize replacing the frozen primary. History sensitivities depend on unverified outcome-maturity assumptions.

## Probability reliability and allocation

Primary ROC-AUC was **0.5996**, average precision **0.3272**, and recorded no-show prevalence **0.2601**. The Brier score was **0.1899**. Raw probabilities overpredicted: their sum was **4,453.7** expected recorded no-shows versus **3,906** observed, observed/expected ratio **0.877**. No final-test recalibration was performed.

The frozen selection concentrated heavily in younger adult age bands, with very little allocation to older adults. This unresolved allocation issue warrants review and establishes neither fairness nor harm. Appointment month was an unseen category for **71.21%** of final appointments; the frozen unknown-category handling was retained.

Only broad cohort totals are distributed in this main-branch Streamlit research release. No age-band counts, subgroup outcome cells, calibration bins, complementary cells, per-date tables, identifiers, row hashes or individual source scores are included in this release.

## Frozen design and independent verification

- 30,888 development appointments, ending 25 May 2016; three expanding temporal folds with maturity gaps.
- Ten registered primary candidates; exactly 40 successful learned fits including fixed history sensitivities and final refits.
- Primary inputs: recorded age, booking lead days and log lead days, appointment weekday, booking weekday, appointment month and hospital location. Identifiers, SMS timing and unverified health-condition fields were excluded from primary predictors.
- Protocol `careaccess-vitoria-temporal-pilot-v1.1`, executable code, model selection, refits and final predictions were frozen before the single recorded final-target parse.
- Procedural target separation was used. This was not independent cryptographic escrow or a blinded prospective trial.
- Independent arithmetic and paired-date resampling checks reproduced the recorded aggregate results. The final rank/probability regressions were not rerun on individual final records; crossed intervals were read from the frozen one-time evaluation.
- The public summary is a minimal, explicit-field extraction of already sanitized aggregate evidence, not a full copy of the private research output. Its SHA-256 is pinned in the read-only display module. No new inference dependency is added.

Intervals use 10,000 paired-date replicates and 10,000 crossed patient/date replicates, with 2,000 date replicates for AUC/AP. Frozen selections and a common weighted slot denominator are retained; there is no refitting or reranking. With only eight dates and two weeks, percentile coverage can be unstable. The crossed interval is a dependence sensitivity, not an exact coverage guarantee. Source-selection, model-development, snapshot and maturity uncertainty are not captured.

## Source, rights and privacy

Retrospective analysis of **Medical Appointment No Shows, version 5**, shared on [Kaggle by Joni Hoppen (joniarroba)](https://www.kaggle.com/datasets/joniarroba/noshowappointments), with credit to **Aquarela Analytics**. Source data: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). CareAccess applied cohort filtering, feature engineering and statistical evaluation. This main-branch Streamlit release shows only aggregate Vitória findings and redistributes no Vitória source records or research model weights. No endorsement by the source contributors is implied.

The frozen aggregate summary's publication-scope statements refer to this aggregate-only Streamlit release. They do not describe a separate branch or browser-demo release, which must be read with its own asset-specific notices and licenses. The MIT software license does not relicense separately distributed dataset-derived assets.

The source-data license does not replace privacy rights. Consent, institutional release authority and anonymization documentation were not verified. The independently written application code remains under its original MIT license; that license does not relicense the source data or private model artifacts. No commercial-use permission is represented.

## Limits and safe interpretation

The printed calendar dates provide a conservative research timing proxy, not verified real-world 48-hour availability. Historical roster snapshots, actual timezone semantics, stability of supplied keys and outcome-finalization timing were not verified. Appointment is the analysis unit, and repeated supplied keys can occur. The source's cancellation and rescheduling rules are undocumented, so its label is not a fully adjudicated clinical outcome.

The short historical window cannot establish annual stability, transportability to current clinics, individual probability reliability, equitable allocation, or outreach effectiveness. Any future operational use needs independent locally governed validation and a prospective evaluation. Ordinary reminders and access to care must remain available to everyone; scores must never be used to penalize people, cancel appointments or deny care.

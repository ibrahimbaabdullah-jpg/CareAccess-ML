# CareAccess predictive-value investigation

13 October 2026. Research result: **do not promote a replacement model. Better predictive prioritization has not been demonstrated.**

## What was tested

A bounded, declared comparison tested 17 learned model configurations and four simple policies on the original Mendeley development data. Two additional historical specialty/calendar rules were declared before results review. The learned configurations covered the original logistic architecture, categorical-calendar logistic, spline logistic, shallow/deeper histogram gradient boosting, random forests, recent-365-day training windows, and basic versus city/disability-augmented inputs.

Each expanding temporal fold trained strictly before its evaluation year: 2017, 2018, and 2019. All imputation, encoding, scaling, spline knots, and historical-rate tables used training data only. The pooled evaluation contains 29,087 appointments, 2,810 missed visits, and 593 appointment days. A fixed once-only 2020 qualification followed model selection. The previously exposed 2021–2022 final cohort was removed before feature/outcome analysis and was never rescored or used for tuning here.

The primary objective was missed visits identified among at most 20 appointments per day, compared at exactly the same daily capacity. The random comparator is the exact expected capture for that day's outcome prevalence, not an arbitrary random seed. A simple historical rule uses smoothed past attendance rates by specialty and age band. It cannot use individual attendance history because the dataset has no patient identifier.

## Primary evidence

The chosen research candidate was a shallow, seven-leaf boosted model with specialty, age, calendar/time, city, and disability. These are exploratory inputs with assumed, not verified, preappointment availability.

| Policy | Development missed visits captured | Precision in 11,228 slots |
|---|---:|---:|
| Matched random expectation | 1,133.99 | 10.10% |
| Historical specialty/age rule | 1,183 | 10.54% |
| Original logistic architecture, temporally refit | 1,192 | 10.62% |
| Selected boosted candidate | 1,247 | 11.11% |

The selected model's additional missed visits identified per 100 outreach slots were:

- Versus matched random: **1.01**, date-cluster 95% interval **0.52 to 1.47**.
- Versus the logistic architecture: **0.49**, interval **−0.20 to 1.19**.
- Versus the historical specialty/age rule: **0.57**, interval **−0.01 to 1.14**.

These are exploratory, selection-biased development intervals, without multiple-comparison correction. The predeclared development gate failed because the lower confidence bound versus the best simple rule did not exceed zero. Better daily ranking than the logistic comparator was also not established. The selected model's pooled ROC-AUC was **0.5221** (date-cluster interval 0.4967–0.5479) and AP **0.0981**, against pooled prevalence **0.0966**. AUC/AP are secondary to the explicitly capacity-matched objective.

## Later-period check

The already-used 2020 calibration cohort provides a later historical check, not a new independent test. It has only 827 appointments and 117 missed visits over 48 appointment days.

At daily K20, 714 appointments were selected:

- Random expected captures: **96.06**
- Selected boosted model: **96**; AUC **0.4821**, AP **0.1415**
- Temporally refit original logistic architecture: **105**
- Historical specialty/age rule: **106**

The selected model's difference versus random was **−0.009 missed visits per 100 outreach slots**, interval **−0.95 to 1.21**. Its development gain did not persist. K20 selected 86.3% of all 2020 visits; 35 of 48 days had 20 or fewer appointments, leaving no prioritization opportunity on those days. Other prespecified K5/K10 summaries are retained in the aggregate results; no capacity was selected after seeing results.

The logistic comparator here is the original architecture refit to each available training boundary. It is **not** a new evaluation of the frozen deployed artifact. Original trained artifacts and historical evaluation are preserved.

## Why performance is weak

The investigation found no target reversal or scheduled-time parsing defect that explains away the result. The mapping is strictly yes=missed and no=attended, scheduled dates use DD/MM/YYYY, time includes minutes, and appointment-time age is derived consistently. The public source does not permit independent adjudication of the underlying attendance labels.

The data nevertheless exhibit substantial drift:

| Year | Appointments | Missed-visit rate | Missing appointment-time age | Missing specialty |
|---|---:|---:|---:|---:|
| 2016 | 7,853 | 4.05% | 0.23% | 4.56% |
| 2017 | 12,129 | 8.41% | 0.31% | 5.78% |
| 2018 | 9,205 | 8.28% | 26.63% | 7.20% |
| 2019 | 7,753 | 13.26% | 35.21% | 20.64% |
| 2020 | 827 | 14.15% | 25.27% | 16.93% |

A larger or more complex learner cannot recover information absent from the source. There is no scheduling creation timestamp, valid patient identifier, versioned feature snapshot, or outreach/outcome history. The provider defines entry-service date as the patient's first visit at the center, not when this appointment was booked. Mixed/invalid dates and entries after appointments further prevent treating it as a reliable booking lead time. It remained excluded. No patient identity was inferred from birth dates or other quasi-identifiers.

City/disability did not establish a useful incremental benefit over the basic boosted model. In a secondary expected-random-tie analysis, their contribution was only 0.08 additional captures per 100 slots, with an interval spanning −0.30 to 0.45. Collecting those extra inputs is not supported by this comparison.

The source is a single historical rehabilitation service. No claim of effectiveness in a different clinic, contemporary population, or patient-independent cohort follows from it. Missed visits identified are not missed visits prevented by outreach.

## Metric integrity and review

The initially valid primary pooled seeded tie policy is preserved exactly. Annual summaries were corrected to slice that same tie vector, rather than restart the generator in each year. Original 2020 ties are also unchanged. This preserves primary counts 1,247 / 1,192 / 1,183 and later counts 96 / 105 / 106.

An alternative tie realization and an exact expected-random-tie policy are explicitly secondary sensitivities. Neither was used to reselect a model or replace the failed primary gate. Tied simple-rule counts vary, but all analyses fail to establish improvement over the logistic comparator and show loss of gain in 2020. An independent reviewer reproduced the original primary counts and all three paired bootstrap intervals, and checked weekly-block sensitivity.

All experiments wrote only to a separate research directory. During this work, a separately authorized release fix changed the manifest's validation-code checksum/revision metadata. This was detected and logged. Model weights, preprocessing artifacts, source fingerprint, training history, and the original evaluation report did not change.

## Computational cost

These are new resource measurements of the research configurations on 36,940 pre-2020 records, not reconstructed telemetry from the original training run. Each candidate ran in an isolated Linux x86-64 process with at most two estimator threads. Peak RSS includes the interpreter, libraries, source parsing, and preprocessing. Inference timing uses only actual-source adult development appointments, privately in memory: the first eligible appointment for single-input timing and the first 1,000 for batch timing. It includes preprocessing; medians use 20 repetitions after warmup. These timing inputs may overlap training and are not predictive validation. Earlier invented timing-only probes were superseded; no invented inputs were used in any predictive fitting or evaluation experiment.

| Configuration | Fit wall time | Process peak RSS | Single input | 1,000 inputs |
|---|---:|---:|---:|---:|
| Historical specialty/age rule | 0.0033 s | 268.0 MiB | 2.84 ms | 2.65 ms |
| Original logistic architecture | 0.0691 s | 272.4 MiB | 2.90 ms | 3.00 ms |
| Selected boosted candidate | 0.5690 s | 283.4 MiB | 4.09 ms | 7.59 ms |

All 21 original configurations have complete resource records. Compute cost is not the limiting factor in demonstrated value.

## Decision and next evidence needed

1. Preserve the current product as an explicitly limited research/workflow workbench. Do not claim validated predictive superiority or automatically act on risk scores as a proven clinical prioritization policy.
2. Stop model fishing on the exposed cohort. This bounded investigation does not prove that no conceivable algorithm could work; it does establish that these defensible alternatives did not satisfy the requested benefit criterion.
3. For a stronger claim, obtain an authorized, genuinely independent dataset with documented prediction time and feature availability. Useful additions include booking timestamp, locally managed pseudonymous patient/appointment IDs, finalized prior attendance/cancellation events known before the scoring cutoff, appointment type/location, and timestamped reminder delivery/response. Do not derive identities from birth dates.
4. Before examining new held-out outcomes, freeze the feature contract, candidate selection, actual daily contact capacity, comparison rules, and a stakeholder-agreed minimum worthwhile increase in missed visits identified. Determine evaluation sample size using the desired margin, paired policy disagreement, event frequency, and day/patient clustering rather than an arbitrary row target.
5. Evaluate once on the untouched future or external holdout, including calibration, same-capacity incremental capture, patient/day-cluster uncertainty, and subgroup performance. A different public dataset can support an external research benchmark; it does not establish benefit for this particular clinic.
6. Demonstrating fewer missed appointments or cost-effectiveness additionally requires a prospective outreach study with an appropriate contemporaneous control and authorization for contact. Prediction performance alone cannot establish that effect.

## Sources and reproducibility

Dataset: Salazar, Luiz Henrique (2023), Medical Appointments No-Show, Mendeley Data V1, DOI [10.17632/wm6w2fvkfj.1](https://data.mendeley.com/datasets/wm6w2fvkfj/1), CC BY 4.0. Source SHA256: 903f30aa008c478dd6451550b53c2bf937653f28763ea288e43f5e99c536cbdf.

The adjacent protocol, source scripts, aggregate research results, completed analysis, resource measurements, tie sensitivities, and validation report preserve this investigation. Raw source records and row-level prediction arrays are private and must not be included in a public report or repository. No experimental model was published or substituted into the application.

## Published evidence links

[Research protocol](../reports/research/protocol.json), [completed primary analysis](../reports/research/completed_analysis.json), [original primary selection](../reports/research/selection.json), [resource measurements](../reports/research/resource_measurements.json), [validation assertions](../reports/research/research_validation.json), [expected-random-tie sensitivity](../reports/research/tie_policy_expectation.json), and [portable reproduction scripts](../scripts/research/README.md). Expected-random-tie sensitivity does not replace the primary failed gate.

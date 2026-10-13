# CareAccess: independent validation review and next-evidence protocol

Reviewed 13 October 2026. This report reviews the current frozen implementation and a bounded follow-on research experiment. It is not a prospective registration, clinical clearance, or proof of benefit.

## Decision

Keep the existing model in research-only status. Do not promote a new candidate or say that CareAccess improves supportive outreach prioritization. The available software can generate real trained-model predictions, but the evidence does not establish useful ranking on later historical appointments, and no intervention study has measured whether its outreach choices help patients attend.

The bounded experiment is worth reporting: it tests realistic alternatives, includes simple policies, and produces a reproducible negative result rather than forcing a successful model. There is no defensible reason to keep searching the same evaluation years until a favorable result appears.

## Verified current-model evidence

The source comprises 49,593 records from 2016–2022. Exact full-row deduplication leaves 46,672. The published frozen evaluation used 8,905 appointments from 2021–2022, including 1,433 missed visits over 231 appointment dates. No patient identifier is available.

Independent inference from the saved artifacts reproduces the published results. At the daily capacity of up to 20 appointments:

- Logistic regression selects 4,420 appointments and captures 697 missed visits. Exact same-day random selection would capture 700.879 in expectation.
- Its precision difference versus random is −0.0878 percentage points. A paired 5,000-date-cluster bootstrap gives a 95% percentile interval of −0.9684 to +0.7801 points. A calendar-week-cluster sensitivity gives −1.1641 to +0.9530 points.
- Every trained model's K5, K10 and K20 interval includes zero. This does not prove exact equivalence to random; it means a reliable gain has not been established.
- Logistic Brier score is 0.135304, versus 0.135403 for a constant prediction using the 2020 calibration prevalence. The originally reported training-era constant baseline scores 0.142941. Most of the apparent Brier improvement over that older baseline is explainable by updating the average event rate, without meaningful person-level ranking.

These intervals are conditional on the frozen predictions and observed selection/tie policy. Dates or weeks are resampled as paired clusters. They do not account fully for repeated patients, model-development uncertainty, or adaptation to this already-seen evaluation set. They are retrospective estimates, not guarantees for a future clinic.

Reproducible files: `check_frozen_operational_metrics.py` and `frozen_operational_check.json` in this review directory. No row-level predictions or patient attributes are published in these files.

## Review of the bounded model-improvement experiment

The research analysis froze a 21-candidate comparison before fitting. It used expanding-time folds: earlier dates train models evaluated in 2017, 2018 and 2019. Candidates included original and richer logistic encodings, splines, several gradient-boosting sizes, random forests, recent-365-day training windows, and basic versus city/disability-augmented inputs. Simple specialty, age-band and specialty-by-age priors and earliest-appointment ordering were included. Records dated 2021 onward were removed before feature/outcome analysis.

Its prespecified primary selection criterion was pooled daily-K20 capture above exact same-day random expectation. The development gate required a positive lower paired-bootstrap bound against random and against the best simple policy, and positive random lift in at least two annual folds. These are research-screening rules, not clinical acceptance criteria, and the selected-candidate intervals remain optimistic because selection and assessment share the same development folds.

Results independently reproduced from the private out-of-fold scores:

- Selected candidate: 7-leaf histogram gradient boosting with city and disability.
- Development: 1,247 missed visits captured in 11,228 slots, versus 1,133.989 expected random, 1,192 from the original logistic architecture, and 1,183 from the specialty-by-age policy.
- Difference versus random: +1.0065 percentage points, date-bootstrap interval +0.5155 to +1.4685.
- Difference versus specialty-by-age: +0.5700 points, interval −0.0089 to +1.1437. The pre-fit gate fails.
- Difference versus original logistic architecture: +0.4898 points, interval −0.1971 to +1.1903.
- A descriptive weekly-cluster sensitivity also includes zero for both of these model-versus-policy differences. No model or gate was changed after this check.
- The selected candidate then underwent the planned one-time 2020 check, using 827 appointments and 117 events across 48 dates. It captured 96 of the missed visits in 714 slots, versus 96.063 expected random. AUC was 0.4821 and average precision 0.1415, essentially equal to the cohort's 0.1415 prevalence. The refitted logistic architecture captured 105; the specialty-by-age baseline captured 106.

A secondary analysis averages uniformly over exactly equal-score cutoff ties rather than relying on one random tie realization. It gives development expected capture of 1,246.598 for the selected candidate, 1,169.015 for specialty-by-age, and 1,189.217 for the logistic architecture. Under this different descriptive estimand, the candidate-versus-simple interval is positive (+0.1342 to +1.2378 percentage points), while the candidate-versus-logistic interval still includes zero (−0.1671 to +1.1961 points). The augmented-versus-basic boosting interval also includes zero (−0.2965 to +0.4479 points). Thus the precise development comparison with a coarse simple policy is tie-sensitive; the original gate is retained, and neither the positive sensitivity nor its sampling interval is independent confirmation. In 2020 the selected candidate's expected capture remains 96, so its later-period failure to establish gain is unchanged. Source: `reports/research/tie_policy_expectation.json`; no reselection was permitted.

2020 is already the original system's calibration cohort. This is an exploratory later-period qualification, not independent final validation. The candidate's development gain did not survive it. Neither the original published 2021–2022 result nor this 2020 check supports replacing the deployed model.

Code inspection found no fitting of preprocessing or group priors on future folds, no forbidden reason/weather/identity/history features, and no new >=2021 evaluation. Original saved model/report hashes were preserved by the experiment. The primary pooled counts and paired-date intervals reproduce to absolute tolerance 1e-12 in `review_development_results.py` and `development_independent_check.json`.

Report accuracy corrections completed and independently checked:

- `original_logistic` is now described as the original architecture refitted for each fold; the 2020 comparator is trained through 2019, unlike the original frozen deployed artifact trained through 2018.
- Annual summaries now slice the original pooled 2017–2019 tie vector. The original primary pooled counts, all three paired intervals, and the original 2020 selected/logistic/simple metrics are exactly preserved. Annual capture totals now sum to the pooled total. An intermediate alternate-tie analysis is preserved only as sensitivity and is not substituted for the primary result.
- The originally promised 500 AUC/AP date-bootstrap replicates have been supplied in `completed_analysis.json`. They remain selected-development descriptive intervals.
- Private timestamp schema metadata now explicitly identifies microseconds.
- Concurrent application integrity work changed only the manifest's configuration-hash/revision metadata during the supplementary analysis. That change was authorized separately; no fitted model, preprocessor or original evaluation metric changed. The audit records the exact hash changes separately. This is not attributed to retraining or hidden by an all-files-unchanged claim.

## What can and cannot be learned from the existing data

Can establish:

1. The source, transformations, real fitting process, software behavior, artifact integrity, and reproducibility of reported retrospective metrics.
2. Whether prespecified alternatives improve development-period ranking, with explicit temporal instability and selection bias.
3. That the current evidence fails to establish a useful improvement over simple allocation policies.

Cannot establish:

1. Fresh independent performance for a newly designed model by reshuffling or reusing the already-examined years. Creating another train/test split, changing a random seed, or naming 2020 a holdout does not restore independence.
2. Performance in new patients: repeat patients may cross years, and no lawful supplied patient key exists to measure or control this.
3. Real decision-time operational value: the CSV lacks versioned scheduling snapshots and booking timestamps. It cannot prove all scored appointments or fields were available when outreach would occur.
4. A causal reduction in missed visits, better access, or savings. Outcome risk is not the same as expected response to outreach. The records contain no randomized supportive-outreach intervention and no adequate intervention log for a causal policy comparison.

## Operational policy mismatch requiring resolution

At this review checkpoint, `careaccess/planner.py` ranks the entire uploaded batch and selects K total, with `appointment_id` as the tie break. `app.py` passes the whole scored batch. The published research metric instead selects up to K for each appointment date and uses a seeded random tie break. These are different policies for a multi-day batch, and can also select different records when scores tie. Consequently, the historical metric does not directly validate the current planner policy. Label it as an appointment-day research scenario, or align and test the intended policy before a future validation. No app changes were made in this review.

## Required data and timing contract for a genuinely new evaluation

Use a newly collected, separately governed cohort from the intended service, or an external provider whose measurements and workflows match the model. Freeze model weights, preprocessing, schema mapping, tie policy, eligibility, decision time, capacity, comparator, primary estimand, statistical analysis, meaningful-effect margin, and stopping plan before outcome access. An independent analyst or locked scoring process should hold outcomes until the evaluation is complete.

For every eligible appointment, a secure healthcare environment should retain:

- A provider-supplied random patient key and a separate appointment key. Do not create identities from birth date, disability, city, or other quasi-identifiers. Patient linkage stays in the provider's approved environment.
- The scheduled date/time, local timezone, clinic/resource, scheduling/creation time, and the actual feature snapshot time. Specialty and age at appointment must be verified as available at that snapshot. Exact birth dates are unnecessary for scoring.
- Any authorized history variables as of the prediction cutoff, using only completed prior appointments and matured outcomes. Never include a current or future outcome in prior-attendance counts. Booking lead time requires a true booking timestamp, not service-entry date.
- The score, model/configuration hashes, eligibility and exclusion reasons, and the actual slots available to staff at that moment.
- A separately matured outcome record distinguishing attendance, no-show, advance cancellation, same-day cancellation, and rescheduling. Define these before the study. Do not silently drop post-snapshot cancellations to make the model look better.
- For a later intervention trial: allocated policy, contact attempts, timing, reachability, channel and language, opt-outs, support offered and accepted, standard reminders, staff effort, and the agreed care-access outcomes.

The prediction cutoff and capacity must reflect the intended workflow, such as a specified number of business days before the appointment. The existing metric is capacity per appointment date; actual outreach staff capacity may be per contact date or clinic. These are different operational constraints and require an explicit prospective definition.

Preserve the leakage contract: no no-show reason, actual appointment-day weather without a historical forecast snapshot, raw retrospectively calculated age/flags, unverified diagnosis timestamps, or fabricated patient-history linkage. Additional city/disability inputs need availability and usefulness evidence; they are not justified solely because one selected historical model uses them.

## Stage 1: prospective silent ranking evaluation

Keep predictions invisible to schedulers so that the model cannot affect observed outcomes during this stage. Maintain ordinary care and reminders.

Primary endpoint:

- At the prespecified capacity K for each operational day/resource, estimate the number of eventual missed visits captured per 100 outreach slots.
- Compare the locked model with both exact matched random-selection expectation and a prespecified practical comparator. The practical comparator should be a locked simple policy or current staff workflow; do not choose it by fresh evaluation outcomes.
- For day d, let m_d=min(K,n_d), h_d be selected missed visits, and r_d be the day's missed-visit proportion. Incremental precision versus random is sum(h_d−m_d*r_d)/sum(m_d). The use of test labels to calculate the comparator expectation is evaluation only, never an inference feature.
- Evaluate only the primary K for confirmation. Other capacities, thresholds, subgroups, and alternative models are secondary or exploratory unless the multiplicity plan explicitly covers them.

Statistical and practical gates:

1. A paired confidence interval for the primary difference against each required comparator has its lower bound above zero. Use patient and operational-cluster dependence in the analysis; consider calendar-block sensitivity for serial correlation.
2. A pre-agreed meaningful improvement is met. The meaningful margin must reflect staff effort, contact cost, patient burden and support availability, not whichever number emerges. For scale only, a +2-point precision gain identifies 20 additional eventual missed visits per 1,000 contacts; it does not mean 20 visits prevented. Two points is an illustration, not an approved target.
3. Prespecified temporal/site consistency checks do not reveal unacceptable deterioration. Define a tolerable subgroup disadvantage with service stakeholders in advance; do not infer fairness from a nonsignificant small-subgroup test.
4. If probabilities will be displayed or used for decisions, also assess calibration-in-the-large, slope/curve, Brier and log loss against an appropriately dated constant-rate baseline. Calibration or updating performed on this cohort creates a new model needing another evaluation cohort.
5. The model meets technical delivery gates: frozen reproducible scoring, finite/valid outputs, strict schema/time validation, correct identity separation, label-free ranking, safe exports, complete aggregate audit evidence, and no operational use outside the approved study.

Also report AUC, AP with the cohort prevalence, capacity fraction, captured counts, uncertainty, expected versus observed events, input missingness, unknown categories, outcome loss, number of patients and operational clusters, and time/site strata. Publish negative and failed gates together with successful ones.

Determine sample size from the desired confidence-interval precision for the primary policy comparison and calibration, accounting for repeated patients, daily workload, intracluster correlation, event prevalence and seasonality. Do not substitute a universal minimum-event rule. Use pre-study simulation and take the largest requirement. Freeze an accrual target plus a sensible administrative maximum; inconclusive results remain inconclusive. Any interim efficacy looks need a sequential-testing plan, rather than repeatedly checking until significance.

## Stage 2: randomized policy-impact trial

A ranking study alone cannot establish that supportive outreach helps. If Stage 1 supports proceeding, compare model-prioritized additional outreach with a practical comparator using the same resource budget and the same support. Keep standard reminders and access to care available in both arms.

Use a randomized allocation-policy design suited to the service. Cluster assignment by clinic/day or clinic/week can avoid staff contamination, but requires enough independent clusters, calendar stratification and control of repeated-patient/carryover effects. A biostatistician should finalize the design rather than treating appointments as independent observations.

The primary analysis should include all eligible appointments under their assigned policy, using intention-to-treat. Comparing outcomes only among selected/contacted people would compare groups chosen for different risk and can give misleading results. Prespecify the primary missed-appointment rate and clinically meaningful absolute reduction. Secondary outcomes should include timely completed care/rescheduling, contact burden, opt-outs, staff effort, costs, and subgroup access; agree the care-completion window with the clinical service.

Report absolute differences and uncertainty, and the contacts/cost required per additional completed visit when estimable. Examine harm and service inequity as well as average benefit. No access denial, punitive action, or automatic overbooking is justified by this model. Use the appropriate local clinical, privacy, and research governance before enrollment.

A published randomized reminder trial shows that intervention effects can be measured, but its estimates cannot be transferred to CareAccess. It randomized an extra reminder within an already high-risk group; it did not demonstrate that CareAccess, or model targeting generally, beats equal-budget random outreach. [Ulloa-Pérez et al., 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9126539/)

## Public-data check

A bounded read-only search did not identify a verified, immediately compatible independent public cohort for this exact model. This is not a claim that none exists.

- The [Mendeley source](https://data.mendeley.com/datasets/wm6w2fvkfj/1) is the same 2016–2022 rehabilitation-center cohort already used.
- The original [Kaggle Medical Appointment No Shows metadata](https://www.kaggle.com/joniarroba/noshowappointments/activity) describes a different feature schema, including scheduling date, appointment date, neighborhood and health flags. Its listed schema does not establish the required specialty and appointment-time equivalence. It would need a documented compatibility audit and potentially a separately developed model; substituting it is not automatic external validation of this model.
- [rdv_maroc](https://data.mendeley.com/datasets/724jjmcyzc/1) supplies too little public description to verify real-world provenance, decision-time availability, or compatible features. It is not yet suitable validation evidence.
- Synthetic appointment datasets can test software and load, but cannot establish real patient-outcome prediction or causal outreach benefit.

No additional patient records were downloaded or replaced in this review.

## Methodology references

The following primary methodology sources support distinct evaluation data, full reporting, calibration/discrimination assessment, and study-specific sample-size planning. The particular CareAccess gates above are proposed design choices, not thresholds mandated by these papers.

- [TRIPOD+AI reporting guidance, BMJ 2024](https://www.bmj.com/content/385/bmj-2023-078378)
- [Evaluation of clinical prediction models, part 2: external validation, BMJ 2024](https://www.bmj.com/content/384/bmj-2023-074820)
- [Evaluation of clinical prediction models, part 3: sample size, BMJ 2024](https://www.bmj.com/content/384/bmj-2023-074821)
- [Riley et al., minimum sample size for external validation, Statistics in Medicine 2021](https://onlinelibrary.wiley.com/doi/10.1002/sim.9025)

## Published evidence links

[Frozen operational metrics](../reports/validation/frozen_operational_check.json), [independent development arithmetic](../reports/validation/development_independent_check.json), [review verification](../reports/validation/review_verification.json), and [portable reproduction scripts](../scripts/research/README.md). Original review hashes identify the reviewed working copies; publication relocation and wording adaptations are recorded separately.

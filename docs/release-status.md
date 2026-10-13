# Release acceptance status

Evidence updated 2026-10-13. [Public application](https://careaccess-ml.streamlit.app/) · [Source repository](https://github.com/ibrahimbaabdullah-jpg/CareAccess-ML)

## Current decision

The security/acceptance repair was merged through [PR #2](https://github.com/ibrahimbaabdullah-jpg/CareAccess-ML/pull/2) at `6c8969251af8e94c33c0ca1fafd336b9f698bfb2`; its exact main tree and CI were verified. The live service subsequently showed the new repair-specific captions and export columns, and provider logs recorded a code pull at 14:28:04 UTC. **The provider does not expose the exact running SHA**, so deployment identity is supported by those observations rather than a runtime commit attestation.

This follow-on revision fixes two navigation issues observed during live checking: chosen outreach capacity reset to 20, and the consent checkbox appeared unchecked after leaving and returning to the batch page while results remained. Capacity and explicitly granted consent now use separate session state that survives widget cleanup; explicit revoke/Clear still removes the uploaded schedule, results and reviews. Legacy/inconsistent batch results without recorded consent are cleared before either Batch or Planner can expose them. Capacity is clamped when a replacement schedule shrinks.

At this document's preparation, the follow-on change is locally verified and awaits its own protected PR/CI/merge and focused live recheck. Consult that PR's final verification record for subsequent execution results. No future hosted result is preclaimed here.

**The predictive-benefit requirement remains unmet.** No replacement model is promoted. Working software, successful exports and security controls do not establish useful clinical prioritization or fewer missed appointments.

## Exact local verification

- Final follow-on suite: **156 passed, zero skipped** with 24 private actual-source adult inputs.
- Public mode: **101 passed, 55 explicitly skipped** because the private source fixture is not distributed.
- Independent review verified the same five changed source/test files and six additional navigation sequences: capacity 0/5/24, review/export and consent persistence across pages, Clear/revoke, and shrinking/empty/replaced schedules.
- Four-model single/batch/reload inference, displayed results and exact server-side download payloads pass. Dependency consistency, compilation and artifact/configuration integrity pass.
- All trained-model/preprocessor bytes, model manifest, original training/evaluation code and historical evaluation outputs are unchanged by this session-state follow-on.

See [capacity/consent verification](../reports/acceptance/capacity-consent-verification.json), [independent navigation checks](../reports/acceptance/navigation-verification.json), and [acceptance matrix](acceptance-matrix.md). The earlier 145-test repair and its historical reports remain evidence for that earlier source snapshot; the new 156-test results supersede intermediate 154-test patch results.

## GitHub and repository controls

The previous repair's [PR CI](https://github.com/ibrahimbaabdullah-jpg/CareAccess-ML/actions/runs/38412748891) and [merged-main CI](https://github.com/ibrahimbaabdullah-jpg/CareAccess-ML/actions/runs/38413601479) passed, each with 95 tests and 50 explicit source-dependent skips. Those counts do not pretend to be CI results for this follow-on revision.

The workflow uses verified full official Action commit pins, requests Contents: read with only GitHub's implicit Metadata: read additionally observed, and disables checkout credential persistence. Main protection was verified through owner settings and API readback: PR required, GitHub Actions `test` required on an up-to-date branch, administrator enforcement on, force pushes/deletion off, and no mandatory independent reviewer. Every follow-on change must use that normal protected path. See [published-release evidence](../reports/published-release-verification.json) and [branch-policy evidence](../reports/security/main-protection-verification.json).

## Hosted evidence for the preceding repair

The [hosted repair snapshot](../reports/acceptance/hosted-repair-verification.json) covers the service consistent with `6c896925`:

- Actual downloaded batch CSVs for Logistic Regression, Gradient Boosting and PyTorch Neural Network match saved-model inference on 24 authorized adult-source appointments, with exact row order and maximum absolute error `8.326672684688674e-17` at tolerance `1e-6`.
- New `input_quality_warnings` export columns and the seeded-random daily-versus-total-batch policy caption are visible.
- Planner capacity-five export, reference/rank order and committed review status passed. Probability error was `5.551115123125783e-17`; committed review edits survived navigation/capacity changes, and capacity zero behaved correctly.
- The navigation capacity/checkbox issues above were observed and are the bounded purpose of this follow-on repair. Its live recheck remains a distinct gate.

The earlier Chromium **Failed - Blocked** attempts were superseded by successful supported browser download-event checks; no browser security settings were changed or bypassed. These later results establish actual downloaded bytes for the tested paths. They do not convert every local test into a hosted check. No raw fixture rows or individual predictions are published.

Earlier all-page rendering and service-reboot recovery remain in [historical verification](../reports/verification.json). The [read-only hosting audit](../reports/security/hosting-audit.json) separately records owner-visible settings/logs and provider-access limits.

## Security and history

A fresh public mirror through the repair merge passed `git fsck --full`. All **7 reachable commits, 129 unique blobs and 490 model-archive entries** were scanned. The only CSV is the zero-row template; ten credential-pattern families found no matches. See [post-merge history inventory](../reports/security/post-merge-history-audit.json) and [security review](security-history-audit.md). This cutoff precedes the present source/documentation follow-on; its final published tree is checked separately.

Bounded CSV parsing, strict input semantics, session clearing and complete artifact/path/type checks remain in place. The patched local dependency audit reports no known advisories among resolved entries, but the exact `torch 2.14.1+cpu` build is still skipped. Full binary provenance, hash-locked installs, complete deployed dependency inventory, provider internals/log retention/backups, unadvertised Git objects and model memorization are not certified. The input schema cannot prove anonymity. Do not upload identifiable or confidential health information.

## Modeling result and remaining evidence

The bounded 21-configuration experiment did not qualify a replacement. Primary development captures were 1,247 selected boosted / 1,192 temporally refit logistic / 1,183 simple specialty-age rule; the once-only, already-used 2020 cohort gave 96 / 105 / 106. Gain did not persist. In the frozen 2021–2022 cohort, logistic daily-K20 capture was 697 versus 700.879 expected under matched random selection; every tested model/capacity gain interval included zero. No reliable superiority or causal outreach benefit is established.

Daily research selection with random tie handling differs from total-batch planning with reference-order ties. Neither metric directly validates the other policy. See [investigation](predictive-value-investigation.md) and [independent review/next-evidence protocol](validation-protocol.md). Actual-source-only resource timing replaces earlier timing probes; no invented cohort was used for model fitting, selection or evaluation.

A genuinely new authorized cohort with documented decision-time features and a frozen policy/statistical plan is required for a stronger predictive claim. Demonstrating that outreach prevents missed appointments requires prospective intervention evidence. Full project acceptance therefore remains false for the requested predictive-benefit goal; software and hosted checks have their own explicitly scoped status.

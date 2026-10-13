# Release acceptance status

Updated 2026-10-13. [Public application](https://careaccess-ml.streamlit.app/) · [Source repository](https://github.com/ibrahimbaabdullah-jpg/CareAccess-ML)

## Decision and status

This revision repairs application acceptance and security gaps and publishes a bounded negative model-improvement investigation. **It does not establish useful predictive prioritization, clinical fitness, or complete hosted acceptance. No replacement model is promoted.**

The new repair source has been checked locally. Publication, exact-commit CI, deployment and final hosted-browser acceptance are distinct gates; do not infer that the public application has this revision merely from this document. The baseline public main at the start of this review was `8d8ea4fe17b4a6b338d567fb8514729c2667a1a1`. Older hosted results below apply to that baseline, not automatically to these fixes.

## Repairs and local verification

- CSV structure, UTF-8, byte/row/column/cell limits are enforced before pandas allocation; semantic checks require the documented timestamp format.
- Clearing a schedule or withdrawing upload authorization clears batch upload/results/reviews and rotates the uploader. This is active-session clearing, not provider-log/backup deletion or clearing unrelated single-entry input.
- Planner review statuses persist through navigation and capacity changes; quality warnings remain in displayed and exported results.
- Bundled artifacts require an exact allowed model mapping, complete digest coverage, safe path containment and code-owned serialized-type allowlists before loading.
- PyArrow is pinned to the provider-reported compatible 24.0.0; packaging tooling was upgraded and local compatibility retested.
- All five model/preprocessor binary files and original evaluation evidence are unchanged. The manifest changes only the validation-code checksum and revision description; original training/data/evaluation/network hashes are preserved.

See [final local release verification](../reports/release-verification.json), [acceptance matrix](acceptance-matrix.md), and [aggregate extended checks](../reports/acceptance/extended_acceptance.json). The extended test inspected exact UTF-8-BOM bytes supplied to download buttons for all four single/batch models and planner capacities. This establishes server-side payload parity locally, **not browser delivery or saved-file integrity on the live service**. Positive checks use 24 private actual-source adult appointments; no row values or individual predictions are published. Source-dependent tests explicitly skip in public CI when that private fixture is absent.

## Predictive-value evidence

The bounded 21-configuration investigation failed its promotion gate. The selected boosted candidate captured 1,247 missed visits in development, versus 1,192 for the temporally refit original logistic architecture and 1,183 for a simple specialty/age rule. Its improvement over the simple rule and logistic comparator was not established. In the once-only 2020 historical check it captured 96, versus 105 and 106, respectively. That already-used calibration year is not a new independent test.

Independent checking of the frozen 2021–2022 results found logistic daily-K20 capture of 697 missed visits, versus 700.879 expected from matched random allocation. All tested model/capacity gain intervals included zero. Daily research capacity and random tie handling differ from the application's total-batch capacity and reference-order ties; the research metrics do not directly validate the batch planner policy. See [investigation](predictive-value-investigation.md) and [independent review and next-evidence protocol](validation-protocol.md).

## Security and history scope

All five baseline commits reachable from advertised remote refs and 60 unique blobs were scanned, including 490 archive entries. No tested credential patterns or raw appointment CSVs were found; the only CSV was the empty template. This is scoped evidence, not proof that no secret or memorized information exists. Unadvertised/orphaned history, forks, backups and provider internals are not covered.

The patched local dependency scan reports no known advisories among resolved packages, with the exact `torch 2.14.1+cpu` build still explicitly skipped. Binary provenance, full deployed inventory, hash-locked reproducibility and platform privacy controls remain gaps. The approved CI workflow now requests only contents-read permission, disables checkout credential persistence, and pins the official checkout/setup-python Actions to verified full commit SHAs. Main-branch protection is handled separately and its final remote state is not asserted by this source revision. See [history and security audit](security-history-audit.md), [read-only baseline hosting audit](../reports/security/hosting-audit.json), and [privacy/security](privacy-security.md).

## Historical hosted checks, before this repair

Recorded 2026-10-13, concluding 13:15:24 UTC:

- Logistic batch inference matched offline inference for 24 authorized adult-source rows, maximum absolute error `2.7755575615628914e-17` at tolerance `1e-6`, from full-precision values visible in the hosted table.
- Planner capacities 20, 0 and 5 and top-five ordering passed; a review edit was visible. Persistence fixes in this revision were not deployed at that time.
- Chromium reported both batch and planner CSV downloads as **Failed - Blocked**. No downloaded-file success was established.
- An authorized reboot produced new server logs at 13:07:49 UTC and recovered. A post-reboot logistic single prediction matched offline exactly; prior batch/planner state cleared.
- All seven pages rendered. Hosted gradient-boosting/neural parity was not established by the logistic check.

Historical [verification record](../reports/verification.json) retains these results and older test/CI counts. The implementation baseline's [CI run 38376426128](https://github.com/ibrahimbaabdullah-jpg/CareAccess-ML/actions/runs/38376426128) passed 23 tests and explicitly skipped 29 source-dependent tests; it is not CI evidence for this repair.

## Follow-on baseline download verification

At 14:15 UTC, the supported browser download event workflow successfully saved actual batch CSV files for Logistic Regression, Gradient Boosting and PyTorch Neural Network. Each export contained the 24 authorized adult-source rows in the expected order; maximum absolute error versus offline saved-model inference was `8.326672684688674e-17`, below `1e-6`. No browser security setting was changed or bypassed. These later observations supersede the earlier blocked-download outcome **for those baseline batch exports**. The baseline planner export was also saved and verified: five rows, exact reference/rank order, probability error `5.551115123125783e-17`, and review-status column present. These checks do not verify the repaired deployment or every export/state-transition path. See [aggregate downloaded-byte evidence](../reports/acceptance/hosted-baseline-download-parity.json). No raw rows or individual probabilities are published.

## Remaining final gates

1. Verify the exact published repair commit and its CI, then verify that the deployment loaded it.
2. Recheck live single/batch numerical parity for all four models, upload validation, clear/revoke behavior, and planner edit persistence on that deployment.
3. Inspect actually delivered browser-download bytes for single, batch and planner exports. A local payload check cannot close the blocked browser-download gate.
4. Reconcile deployed dependency inventory and authorized repository/platform security controls, keeping unavailable provider controls and exact CPU-wheel advisory coverage explicit.
5. For any superiority claim, obtain a genuinely new authorized cohort and freeze decision-time features, actual capacity, policy/ties and a meaningful-effect criterion before outcomes. Outreach effectiveness requires a prospective intervention study.

Full release acceptance remains false while these gates are open. See [deployment and rollback instructions](deployment.md).

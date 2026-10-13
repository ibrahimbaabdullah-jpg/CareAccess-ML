# CareAccess ML security audit

Audited: 2026-10-13 UTC. Baseline main: `8d8ea4fe17b4a6b338d567fb8514729c2667a1a1`.
Repository: https://github.com/ibrahimbaabdullah-jpg/CareAccess-ML

This is a scoped engineering audit, not a penetration-test certification or a guarantee of security. No repository security settings, credentials, history, live service state, or application source were changed by this audit. Proposed fixes are handled separately and must be verified against their final commit.

## Scope and results

A public `git clone --mirror` completed successfully, with `git fsck --full` passing. Every commit reachable from the server-advertised refs was inspected: **5 commits, 60 unique file blobs, 1,367,896 unique blob bytes**, two branches (`main`, `development/first-release`) and PR #1 head. There are no advertised tags. The releases endpoint returned an empty list. All six accessible GitHub Actions runs returned empty artifact lists (zero retained run artifacts). This includes previous accessible revisions, rather than only today's tree.

The scan covered every blob, all five commit messages/metadata objects, and all **490 entries** inside four skops model archives, including arrays and schemas. Ten credential-pattern families found **zero matches** (private keys, GitHub tokens, AWS keys, Google keys, API-key families, Slack tokens, JWTs, database URLs with passwords, literal secret assignments, long Bearer credentials). This does not prove there are no secrets in unrecognized formats. The only CSV anywhere in reachable history is the **zero-row, four-column template**. Static examination found no raw appointment dataset, patient mapping, private source test fixture, database dump, executable/ELF/PE binary or pickle artifact. Model archives contain numeric or non-object string arrays; no NPY object-dtype arrays were found. Neural safetensors metadata describes only six F32 weight/bias arrays. Models can still retain information statistically; no membership-inference or differential-privacy guarantee was tested or claimed.

Not covered by a mirror: deleted/unadvertised branches, orphaned remote objects, forks, deleted release assets, GitHub/provider backups, organization/account controls, credential history, provider internals or inaccessible private artifacts. Public commit metadata and author attribution are not treated as leaks.

## Prioritized findings

### SEC-01. Medium: malformed CSVs can consume disproportionate resources before schema validation

`app.py:164` calls pandas before checking the exact header or bounding columns/cell lengths. The 5 MB and 5,000-row limits do not bound parser complexity for extremely wide CSVs. A **local-only**, resource-limited adversarial test with 100,000 duplicate columns was only 400,000 bytes; pandas did not finish within its 8 CPU-second test limit and the child was killed. No live endpoint was load-tested.

Fix: validate UTF-8, header, duplicate/extra columns, row width, per-cell lengths, byte limit, and row count before constructing a DataFrame. Keep errors generic and fail the whole file. Add source-independent negative tests, including wide headers, NUL bytes, invalid encodings and malformed/oversized cells. This reduces a demonstrated resource-amplification path but is not a full rate-limiting or denial-of-service defense.

### SEC-02. Medium: main has no branch enforcement

The branches endpoint reports `protected: false` for both branches; the repository rulesets endpoint returns `[]`. The branch-protection detail endpoint returns HTTP 403 because this integration lacks administration-read permission. The observed branch metadata is evidence of absent protection; the detail permission gap must not be described as full settings access.

Suggested separately approved change: require a PR to main, require the unique GitHub Actions `test` check and an up-to-date branch, block force pushes/deletion, and enforce the rule for administrators. Zero mandatory independent reviews avoids locking a solo maintainer out. Consequence: direct main pushes stop working and merges must wait for CI. These controls are supported on public repositories on GitHub Free. Do not change without action-time authorization.

### SEC-03. Medium: dependency reproducibility and advisory coverage gaps

The fresh installed-environment audit inspected 81 package entries. It reports **16 advisory entries, representing 8 unique advisory IDs**, affecting local `pip 25.0.1` (6 IDs) and `setuptools 78.1.0` (2 IDs). These concern packaging/install/build paths; they are not evidence of exploitable unauthenticated app endpoints. Some have additional platform/version preconditions. `requirements-lock.txt` records affected setuptools 78.1.0. CI does not install that full lock; direct runtime dependencies are pinned but transitive dependencies resolve afresh. There are no package hashes.

Exact `torch 2.14.1+cpu` was skipped by the PyPI-based scanner. Fresh supplemental lookups for base `torch 2.14.1`, reported deployed `pyarrow 24.0.0`, `pip 26.2.1`, and `setuptools 83.0.0` reported no known advisories. A base-version lookup is **not** verification of the exact CPU wheel or its provenance. CI currently upgrades tooling; latest logs show setuptools 84.0.0 installed. The local virtual environment, CI runner and hosted environment are separate.

Hosting review reports Python 3.12.15 / Streamlit 1.65.0 and provider replacement of pyarrow 25.0.1 with 24.0.0 for a segfault workaround. This establishes runtime drift from the recorded lock, not by itself a vulnerability. Verify the exact full deployed dependency inventory before calling it audited. Avoid arbitrary runtime upgrades until model compatibility and hosted checks pass.

### SEC-04. Low/medium: Clear schedule leaves the uploaded CSV in the session

`app.py:191-193` removes `batch_scores` but leaves the file-uploader contents and the unrelated `last_single_prediction` session entry. The displayed action says “Clear schedule and results,” but the schedule can still be parsed and displayed on the next rerun. This is within-session retention, not demonstrated cross-user leakage.

Fix: reset uploader state through a supported key/version mechanism and remove associated batch results. Provide a clear all-inputs/results action if single-entry inputs are also meant to be cleared. Describe limits accurately: removal from active app state cannot guarantee deletion from browser downloads, hosting backups, or technical logs.

### SEC-05. Low: model integrity checks do not require coverage of every loaded artifact

`careaccess/inference.py:18-33` validates only keys present under manifest `sha256`. It does not require all paths referenced in `model_files` to be basenames, present in the hash mapping, or members of a fixed supported artifact set. Removing a hash entry can leave that file unchecked. Current committed paths and hashes are valid, and no user-supplied model upload/path is exposed. Exploitation requires repository/deployment tampering or a malformed future manifest; this is defense-in-depth, not an exposed model-deserialization endpoint.

Fix: validate manifest structure, exact supported paths, complete digest coverage, sizes/symlinks as appropriate, and explicitly audited untrusted-type names before deserialization. Continue safetensors for torch. Hashes and manifest in the same repository cannot protect against a malicious maintainer changing both.

### SEC-06. Low: CI supply-chain hardening is implicit

Workflow uses mutable `actions/checkout@v4` and `actions/setup-python@v5`; no explicit `permissions` block; checkout persists the token by default. Latest run's actual token grants are **Contents: read, Metadata: read, Packages: read**, so there is no evidence this job currently receives write access. No `pull_request_target` or untrusted interpolation was found. No deployment or artifact-upload action exists in this workflow.

Suggested changes: pin each Action to verified full commit SHA, explicitly set `permissions: contents: read`, and `persist-credentials: false`. The latter security/permission changes await specific approval. Verified at audit time and matching the latest CI logs: checkout v4 = `11d5960a326750d5838078e36cf38b85af677262`; setup-python v5 = `a26af69be951a213d495a4c3e4e4022e16d87065`. Pinning requires future deliberate updates.

### SEC-07. Low: key negative-input tests depend on a private fixture and are skipped in CI

Existing age/range/forbidden-column/duplicate-reference tests in `tests/test_core.py` require a private actual-source CSV. Latest successful main CI run reports **23 passed, 29 skipped**. Security regressions should have minimal source-independent adversarial fixtures so these checks run on every pull request, while actual-source numerical parity remains explicitly skipped if source authorization/data are unavailable.

## Controls reviewed positively

- Exact four-column input schema; age limits, nonidentifying-token syntax, timestamp/specialty validation and duplicate-reference rejection.
- No upload execution, uploaded model loader, shell invocation, SQL query, arbitrary network fetch, or app-driven filesystem write found in request paths.
- Static HTML styling only; user fields are rendered using Streamlit data components, not concatenated into unsafe HTML.
- Spreadsheet formula escaping is present. Accepted model input fields are strongly constrained; test broader control-character/formula cases as regression coverage.
- Public app warns against identifiable/confidential data and requires an authorization attestation. These cannot prove anonymization or regulatory compliance.
- Uploaded results are in per-session state, not a global input/result cache. Global `cache_resource` contains only the model bundle/aggregate report. Inference reads shared models and does not intentionally fit/mutate them.
- Bundled artifacts use skops plus explicit checks and safetensors, rather than arbitrary pickle execution.
- Streamlit upload limit configured at 5 MB; usage telemetry disabled. No deliberate patient logging or database persistence found.
- Source acquisition requires output outside the repository and verifies the source SHA256; source data is not included in repository history.

## Access and operational limits

GitHub collaborator listing and Actions-settings endpoints are unsupported by the connector; branch-protection details are forbidden to the integration. Repository metadata's owner-admin flag does not grant the connector those capabilities. No collaborator roster, account MFA, deploy key/OAuth scope, secret-scanning settings, GitHub secret contents, provider logging/retention policies, or backup deletion was verified by this repository audit. Hosting review is separate and should carry its own evidence.

No active authentication bypass, cross-user exploit, credential use, destructive test, real patient upload, or live denial-of-service test was attempted. Public accessibility is an intended hosting choice, not classified here as a vulnerability. Rate limits and provider session/download isolation remain provider/runtime controls with incomplete verification.

## Reproduction and evidence

- `audit_history.py`: read-only scanner; `history-audit.json`: full sanitized blob/archive inventory and pattern scan.
- `git-object-inventory.txt` / `advertised-refs.txt`: objects and mirrored server refs.
- `probe_csv_bounds.py`: bounded local wide-header regression probe. CPU hard limit intentionally terminates the original parser; do not run against the public service.
- `fresh-dependency-audit.json`: current local installed environment; `supplemental-dependency-audit.json`: version-only supplemental lookup, not full deployed inventory.
- Main CI: https://github.com/ibrahimbaabdullah-jpg/CareAccess-ML/actions/runs/38403940343
- Branch list: https://api.github.com/repos/ibrahimbaabdullah-jpg/CareAccess-ML/branches?per_page=100
- Rulesets: https://api.github.com/repos/ibrahimbaabdullah-jpg/CareAccess-ML/rulesets?per_page=100
- GitHub protected-branch availability: https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches
- GitHub secure Actions use: https://docs.github.com/en/actions/reference/security/secure-use
- Streamlit cache/session separation: https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource
- Streamlit uploaded-file memory behavior: https://docs.streamlit.io/knowledge-base/using-streamlit/where-file-uploader-store-when-deleted

## Acceptance checklist for remediation

Re-run security tests without private source data, full applicable suite with authorized private fixture if available, artifact hashes/manifest validation, and release checks. Verify no sensitive fixtures entered the commit. Verify published commit and its CI before claiming deployed remediation. Repeat hosted clear/upload/download/session checks against that commit. Record branch-policy changes separately with their authorization and verified final state. Remaining provider controls and binary-advisory gaps must stay visible.

## Local remediation review, 14:00 UTC

The separately prepared candidate was independently reviewed after the baseline audit. This is **local-only acceptance**, not evidence that fixes are published or deployed. `candidate-file-hashes.json` records the exact reviewed source/test/dependency-file snapshot.

- SEC-01: bounded header/row/cell parser and adversarial tests address the demonstrated CSV resource-amplification path. The same 400,000-byte wide-header probe rejects in 0.00186 seconds before DataFrame allocation.
- SEC-04: batch reset rotates the upload widget, clears scores/review state, and revokes the batch attestation. Caption now accurately scopes clearing to batch state; no claim to clear single-entry input or hosting records is made. Source callbacks and unit tests pass; verify uploader disposal and reset in the hosted browser.
- SEC-05: required model mapping, complete checksum set, path containment, digest format and code-owned exact skops-type allowlist fail closed before deserialization. Negative integrity tests pass.
- SEC-07: added source-independent semantic-negative tests and structural-parser/security tests run without a private source fixture.
- SEC-03: local packaging tools are now pip 26.2.1 / setuptools 83.0.0 and pyarrow is pinned to the provider-compatible 24.0.0. The patched local audit reports zero known advisories, **with the exact torch 2.14.1+cpu build still explicitly skipped**. This does not resolve binary provenance, deployed inventory or reproducible hashed-install coverage.
- SEC-02 and SEC-06: branch and CI permission changes remain separately approval-gated. The workflow had not changed at this review snapshot.

Independent checks: **85 passed, 30 explicitly skipped** in 5.92 seconds without a private source fixture; `pip check` passed; release integrity checks passed. Private actual-source parity explicitly did not run in this independent check. All five model artifact hashes, original training configuration hashes and evaluation JSON remained identical to baseline. Final commit CI and hosted acceptance remain required before declaring release completion.

## Published evidence links

[Baseline reachable-history inventory](../reports/security/history-audit.json), [baseline local dependency scan](../reports/security/fresh-dependency-audit.json), [supplemental version lookups](../reports/security/supplemental-dependency-audit.json), [patched local dependency scan](../reports/security/dependency-audit-patched.json), [reviewed candidate hashes](../reports/security/candidate-file-hashes.json), [baseline inference parity](../reports/security/baseline-parity.json), [final release verification](../reports/release-verification.json) and [read-only history scanner](../scripts/audit_public_history.py). Baseline observations above are not automatically claims about a subsequently deployed repair commit.

## Approved workflow hardening in the publication revision

After the baseline and local-remediation review above, explicit authorization was given for CI least privilege. The publication workflow sets `permissions: contents: read` (unspecified optional token permissions are disabled), configures checkout `persist-credentials: false`, and pins checkout v4 to `11d5960a326750d5838078e36cf38b85af677262` and setup-python v5 to `a26af69be951a213d495a4c3e4e4022e16d87065`. Both pins were reverified against the official repositories at 14:17 UTC. Main-branch protection is configured through a separate authorized owner-settings operation; its final remote state is not claimed by this source review. Exact-commit CI must verify the modified workflow.

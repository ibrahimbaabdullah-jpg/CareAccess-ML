# Privacy and security

Only original code, aggregate evidence, compact fitted artifacts and a header-only CSV template are published. Raw appointment records, birth dates, private positive-test inputs and individual prediction vectors are excluded. A public dataset license is not permission to redistribute identifiable patient information. Do not upload identifiers or confidential health information to this public application.

## Data handling

The app requires confirmation that inputs are nonidentifying and authorized for processing on the hosting service. Its strict field allowlist and reference syntax cannot prove anonymity; appointment time, specialty and age can identify someone in context. This is not a healthcare-compliance or de-identification certification. Childhood cases occur in the original data; only aggregate analyses are published and sensitive childhood records must not be uploaded.

Uploaded files are processed in active session memory without an application database or intentional patient-data logging. Streamlit and the hosting provider nevertheless process uploaded bytes. No promise is made about provider technical logs, backups or retention outside application control.

Changing the selected file or model preserves the explicitly labeled last successfully scored schedule until a new scoring attempt. Every new scoring attempt clears prior batch results and review edits before validation. Any invalid row blocks the entire file. Explicit consent persists within the session across page navigation until revoked or cleared; consent is never granted by default. A legacy/inconsistent result without recorded consent is cleared before Batch or Planner can expose it. Clearing the schedule or withdrawing authorization removes the batch upload, batch scores, capacity and planner review state, revokes the confirmation and rotates the uploader. It does not erase downloaded copies, provider logs/backups or unrelated single-entry inputs.

## Input and model controls

A bounded UTF-8 parser checks the 5 MB limit, 5,000-row limit, exact unique four-column header, physical line width, 128-character cells, row shape, malformed quoting and NUL bytes before constructing a DataFrame. Semantic validation checks the exact zero-padded timestamp format, age bounds, specialties and duplicate references. These limits reduce a demonstrated resource-amplification path; they are not comprehensive denial-of-service protection or rate limiting.

Spreadsheet formulas are escaped on export. No uploaded code is executed, no uploaded model is accepted, and no automatic calls, SMS, cancellations, penalties or eligibility decisions are performed.

Model loading requires fixed artifact names, safe non-symlink paths, complete SHA256 coverage and code-owned permitted serialized types. Neural weights use safetensors. Checksums stored beside models cannot defend against a maintainer changing both. Models are not formally differentially private and membership inference was not tested.

## Dependency and history evidence

The original dependency scan in `reports/dependency-audit.json` found packaging-tool advisories. It is retained as historical evidence. After pip 26.2.1 and setuptools 83.0.0 upgrades, the new local scan in `reports/security/dependency-audit-patched.json` reported no known advisories among resolvable entries. **Exact torch 2.14.1+cpu remains skipped by the PyPI-based scanner.** Base-version advisory lookup is not verification of that binary or its provenance.

PyArrow is pinned to 24.0.0, matching the provider-reported compatibility workaround and local tested environment. Runtime direct dependencies are pinned, but the full lock is an environment record and is not the deployment install mechanism. Transitive resolution, unhashed packages, provider modifications and the actual deployed inventory remain separate limitations. No vulnerability-free claim is made.

[The scoped history audit](security-history-audit.md) inspected every commit reachable from the baseline's advertised refs and archives, finding no tested credential-pattern matches or raw-data CSVs. It does not cover unadvertised/deleted objects, forks, provider backups, account MFA, inaccessible secrets or platform internals. The approved workflow now explicitly requests contents-read permission, disables checkout credential persistence and pins official Actions to verified full commit SHAs. Main-branch enforcement is now verified: pull request and up-to-date test check required, including administrators, with force pushes/deletion blocked and no mandatory independent reviewer. See reports/published-release-verification.json for scope.

See [release status](release-status.md), [security history evidence](../reports/security/history-audit.json) and [acceptance matrix](acceptance-matrix.md). Software tests, model metrics and hosted security are separate. No security guarantee or clinical-effectiveness claim is provided.

## Public hosting boundary clarified for the research release
This public Streamlit demo must not receive or process sensitive personal information, including patient-level health information. Do not upload patient records or enter patient health details. Public availability, removing names, or checking a consent box does not establish that data is suitable for this host. Source-data evaluation is performed offline. Real-patient use requires a separately governed deployment and review.

See [Streamlit Community Cloud Addendum 4](https://streamlit.io/terms-of-use). Local or institutional deployment still needs an appropriate privacy, security, legal and clinical review. The checkbox is an acknowledgement, not an override of these restrictions. Historical hosted verification records are preserved; new positive source-derived inference checks run offline only. The Vitória release distributes broad aggregate findings without source records, individual scores or model weights.

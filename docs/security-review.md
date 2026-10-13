# Security and privacy review

Reviewed 2026-10-13 against the local no-generated-data release candidate. This is a bounded engineering review, not a penetration test, legal opinion, de-identification certification or guarantee of security.

This document preserves the scope and findings of that local review. Later integration and hosted-browser evidence are recorded in [release acceptance status](release-status.md) and [machine-readable verification](../reports/verification.json); they do not extend this review into a security certification of the public service.

## File and secret-pattern scan

Scanned 48 candidate files (1,324,679 bytes before this review document), excluding `.venv`, `.git`, Python caches and pytest caches. Also inspected the contents of 490 archive entries inside the bundled `.skops` artifacts. The scan found **zero matches** for the tested private-key, GitHub token, AWS access-key, Google API-key, OpenAI/Groq-key and long literal Bearer-token patterns. This pattern scan can miss other secret formats and does not audit remote Git history or hosting secrets.

Only one CSV exists in the candidate release: `examples/appointments_template.csv`, with headers and **zero data rows**. The earlier generated appointment examples are absent. No original source CSV, private positive-test fixture, birth-date table, patient identifier mapping or exported record-level prediction array was found in the candidate files. Evaluation reports contain aggregate statistics and curves rather than source appointments. The original data and actual-record test fixture remain outside the repository. Their absence from this candidate is not proof of their absence from every previous remote commit; the publication owner should verify the exact remote tree and history before changing repository visibility.

## Model loading

Inference loads bundled model files only. The application does not accept uploaded model artifacts. SHA256 checks cover the committed model artifacts; `.skops` loading compares untrusted types with the committed manifest and rejects types outside the permitted scikit-learn/NumPy namespaces. PyTorch weights use `safetensors`, followed by loading into the known network architecture. No arbitrary uploaded pickle execution is part of the workflow.

Checksums protect against unintended file changes, not a malicious maintainer changing both a model and its manifest. The reviewed manifest points to the expected local artifact basenames. These controls require repository and deployment configuration integrity.

## User-input controls

CSV handling limits uploads to 5 MB and 5,000 rows, checks the exact column allowlist, validates numeric ranges and date formats, rejects unknown specialties and duplicate appointment references, and does not execute uploaded code. Spreadsheet formula prefixes are escaped on CSV export. Invalid schedules are not silently scored. Inputs are session-held, not deliberately persisted in a database or appended to training data.

**The schema and reference regex are not an anonymization system.** A syntactically valid appointment reference could still contain a person's name or a real medical-record identifier. A combination of appointment time, specialty and age may also be identifying in context. The application therefore requires user authorization/absence-of-identifiers attestation and warns against identifiable or confidential records; those notices do not establish a healthcare-compliant hosting arrangement. No real source rows should be copied into the public deployment as examples. Hosting-platform logs and retention require separate review.

## Dependency audit and remaining gaps

`reports/dependency-audit.json` reports advisories affecting the inspected local **pip 25.0.1** and **setuptools 78.1.0** tooling. This is not an all-clear dependency scan. The audit also **skipped PyTorch 2.14.1+cpu** because that exact build was not found by its PyPI lookup; PyTorch advisory coverage is consequently incomplete. A separate base-version lookup in `reports/torch-base-audit.json` reports no known advisories for torch 2.14.1. This narrows the gap but is not verification of the exact `+cpu` binary artifact or a security certification. Other audited dependency entries had no advisories in that particular report, which does not guarantee absence of vulnerabilities.

The release verification record separately reports successful upgrading of pip/setuptools in GitHub Actions. That CI environment is distinct from the inspected local audit environment; its success does not retroactively remove local findings. Confirming the deployed tooling/dependency versions and reviewing the skipped PyTorch build against the vendor's advisories remain separate verification gaps. Do not report that all dependencies are vulnerability-free.

## Verification evidence and scope

The revised release verification record reports 52 local tests using 24 private actual-source adult appointments, and GitHub Actions reports 23 passing tests with 29 source-dependent tests explicitly skipped. The private rows and individual probabilities are not distributed; aggregate parity errors and tolerances are recorded in `reports/inference-verification.json`. Those CI skips must remain visible rather than being counted as passes. Earlier independent local source-preparation, chronology, artifact loading, input-validation and CSV-export checks passed; this review adds file/archive and privacy inspection of the revised candidate.

The final-test discrimination remains approximately chance. A prominent application warning correctly states that effective operational outreach targeting is not established. No clinical approval or intervention-effectiveness claim is supported.

**Scope still not certified by this review:** exact published-tree/history inspection, deployed dependency versions, hosting-platform privacy/retention controls, and a complete hosted security review. Hosted functional checks, downloads, restart behavior and numerical parity are tracked individually in the current [release acceptance status](release-status.md). A passing functional check does not establish clinical fitness, anonymity, or security of the hosting environment.

# Privacy and security

Only original code, aggregate evidence, fitted compact artifacts and a header-only CSV template may be published. No original records, dates of birth, city/ICD/disability row-level values, patient identifiers, raw uploads or test-row prediction vectors are shipped. Public users must confirm authorization for nonidentifying data processing. The field allowlist rejects unexpected columns but cannot prove data are anonymous. Do not upload identifiable health information.

No API keys needed. No patient data logging. Uploaded CSV bytes parsed as data; not executed. 5MB/5000-row limits and exact schema prevent arbitrary large/model uploads. Outputs escape spreadsheet formula prefixes. Uploaded inputs are session-memory-only without intentional persistence; cloud provider processing still occurs. Reset/new-file flows invalidate prior outputs.

Model files are maintained in the reviewed repository only, verified against SHA256 and permitted serialized types. Neural weights use safetensors, not arbitrary pickle. The trust boundary includes repository maintainers: hashes in the same repository do not protect against a malicious maintainer altering both artifact and manifest. No user-supplied model paths are exposed.

Childhood cases occur in source records. Only aggregate analyses are published; this is not a clinical service or HIPAA-compliance claim. Public dataset licensing does not eliminate data protection obligations. Models themselves are not formally differentially private; no guarantee against membership inference is claimed. Training minimized inputs; no direct identifiers or diagnoses in model features.

Security scan results and hosting verification are tracked separately. No warranty or security certification.

## Dependency audit (2026-10-13)
Installed-environment pip-audit completed: 16 advisory entries across pip25.0.1 and setuptools78.1.0, including duplicated registry entries. These are packaging/build tools; no advisories were reported for installed runtime ML/web dependencies in that scan. This is not a guarantee of security. Advisory identifiers and available fixes are in reports/dependency-audit.json. Official-package upgrade attempted but proxy403 and cancelled approval prevented completion; local tooling remains affected. CI requests upgraded packaging tools before installing runtime requirements.

The installed-environment advisory scanner could not resolve the exact `torch==2.14.1+cpu` wheel. A separate lookup for base `torch==2.14.1` reported no known advisories (reports/torch-base-audit.json). This is partial advisory coverage, not an audit or security certification of the CPU binary. Input reference syntax also cannot prove anonymity: users remain responsible for authorization and de-identification.

# Security and privacy review

Updated 2026-10-13. This is a scoped engineering assessment, not a penetration-test certification or a guarantee of security.

The [full baseline history and remediation review](security-history-audit.md) records the exact scope, findings, sources and local repair checks. It supersedes the earlier candidate-only review; historical dependency reports remain available and are not silently rewritten.

## Verified repairs

- Bounded structural CSV parsing rejects resource-amplifying headers/rows before pandas allocation.
- Batch upload/results/reviews clear when the user clears the schedule or withdraws upload authorization; scope is active batch session state.
- Complete artifact mapping/checksum/path/type validation fails closed before loading.
- Negative security tests run without a private patient fixture. Additional positive inference and UI checks use private actual-source adults and explicitly skip on public CI.
- Local dependency/tool compatibility was retested after PyArrow 24.0.0 and packaging-tool changes. No fitted-model or frozen evaluation bytes changed.

The independent final local acceptance suite passed 145 tests; public mode passed 95 and explicitly skipped 50 source-dependent tests. These do not establish browser download delivery, deployed fixes or third-party platform controls. Exact source hashes and final staged verification are in [release verification](../reports/release-verification.json).

## Important remaining limits

The approved workflow now explicitly requests contents-read permission, disables checkout credential persistence and pins official Actions to verified full commit SHAs. Main-branch protection is configured separately; its final remote state is not certified by this local review. Exact CPU-wheel advisory/provenance coverage, full deployed dependency inventory, provider logs/retention/backups, account controls and unadvertised Git objects are not completely verified. Pattern-based secret scans can miss unknown formats, and model memorization is not excluded.

The four-column schema cannot certify anonymity. No raw source rows, private fixtures, per-record prediction arrays or identifiable health information may be published. Local download-payload parity does not establish that the browser saved matching files. See [privacy/security](privacy-security.md) and [release acceptance](release-status.md) for the current distinctions and open gates.

# Release acceptance status

Updated 2026-10-13. [Public application](https://careaccess-ml.streamlit.app/) · [Source repository](https://github.com/ibrahimbaabdullah-jpg/CareAccess-ML)

## Release summary

The application is merged into public `main` and deployed on Streamlit Community Cloud. Hosted logistic batch inference, the outreach planner, and service restart recovery have passed the checks below. Full browser acceptance remains incomplete: the browser blocked both attempted CSV downloads, so downloaded-file contents have not been verified.

This is a retrospective evaluation workbench. The models discriminate near chance, and useful operational prioritization has not been established. Public availability and passing software checks do not establish clinical fitness, effective outreach targeting, or an approved environment for confidential patient data.

## Implemented

Original-data acquisition and audit; four actually trained models; frozen chronological evaluation; saved-artifact loading; seven Streamlit pages; model cards; attribution and reproducibility documentation. Only a header-only CSV template is distributed. No generated patient examples, original records, private test inputs, or individual test predictions are published.

## Verified locally: historical evidence

52 pytest tests passed in 7.77 seconds using 24 private, actual adult-source appointments. All four models passed batch/single/reload parity within absolute tolerance `1e-6`. Streamlit AppTest also verified all three candidate models against offline inference. Only aggregate verification results are published. These historical checks were not rerun for this documentation-only update; they must not be represented as new hosted tests.

## Verified on GitHub

The implemented application was merged at `ad3490293495d6756397524081032171608d4be7`. Its [GitHub Actions run 38376426128](https://github.com/ibrahimbaabdullah-jpg/CareAccess-ML/actions/runs/38376426128) succeeded and completed at 09:28:30 UTC on 2026-10-13.

23 tests passed and 29 source-dependent tests were explicitly skipped in 4.41 seconds because the private fixture was not distributed to the runner. Dependency consistency and bundled model/configuration integrity checks passed. CI successfully installed pip 26.2.1 and setuptools 84.0.0; this does not resolve findings in the separate local audit environment or prove the deployed environment has the same versions.

The earlier [run 38374976468](https://github.com/ibrahimbaabdullah-jpg/CareAccess-ML/actions/runs/38374976468), for commit `95afe10b7b00aea86910936f7c9177706d5bd6f2`, also passed 23 tests with 29 explicit skips in 7.53 seconds. See [machine-readable verification](../reports/verification.json) for scope and [privacy/security](privacy-security.md) for audit limitations.

## Hosted-browser verification

- The live application accepted and scored 24 authorized, private actual-source adult appointments using Logistic Regression.
- All 24 full-precision probabilities were read from the hosted table after choosing **Format > Plain** and compared with offline inference. Maximum absolute error was `2.7755575615628914e-17`, below tolerance `1e-6`. This verifies the observed UI values; it is not a downloaded-byte check.
- The planner respected capacities of 20, 0, and 5. Its top-five ordering matched the offline result, and a review-status edit was visible.
- Batch and planner CSV download attempts both appeared in Chromium's download history as **Failed - Blocked**. No downloaded bytes were available to inspect; no export success is claimed.
- An actual service reboot produced new startup logs at 13:07:49 UTC, and the app recovered. A post-reboot single-appointment UI prediction matched offline inference exactly (absolute error `0`).
- All seven page routes rendered: Home / Dashboard, Live Prediction, Batch Prediction, Outreach Planner, Model Comparison, Model Evaluation, and About / Documentation.
- The reboot cleared the previous batch and planner session state. Hosted checks concluded at 13:15:24 UTC on 2026-10-13.

## Behavior and remaining limits

The entire input file must pass validation before batch scoring; valid rows are not selectively scored. Changing the selected file or model preserves the clearly labeled last successful results until a new scoring attempt or **Clear schedule and results**. Starting a scoring attempt clears previous batch results before validation.

Hosted neural-network and gradient-boosting numerical parity are not established by the logistic batch check. The local security review does not certify the complete published Git history, hosting-platform privacy/retention controls, or deployed dependency versions; advisory coverage of the exact PyTorch CPU binary remains incomplete. These limits and the blocked download checks remain explicit release evidence gaps. No full-release, vulnerability-free, or clinical-validation claim is made.

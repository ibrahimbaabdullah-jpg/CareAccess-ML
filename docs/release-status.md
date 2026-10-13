# Release acceptance status

Updated 2026-10-13.

## Implemented
Original-data acquisition and audit; four actually trained models; frozen chronological evaluation; saved-artifact loading; seven functional Streamlit pages; model cards; attribution and reproducibility documentation. The latest revision removes generated patient examples. Only a header-only CSV template is distributed.

## Verified locally
52 pytest tests passed in 7.77 seconds using 24 private, actual adult-source appointments. All four models passed batch/single/reload parity within absolute tolerance 1e-6. Streamlit AppTest also verified all three candidate models against offline inference. Only aggregate verification results are published; neither the input records nor their predictions are distributed. Independent methodology and privacy checks passed.

## Verified on GitHub
Commit `95afe10b7b00aea86910936f7c9177706d5bd6f2`: [GitHub Actions run 38374976468](https://github.com/ibrahimbaabdullah-jpg/CareAccess-ML/actions/runs/38374976468) succeeded.

23 tests passed and 29 source-dependent tests were explicitly skipped in 7.53 seconds because the private fixture was not distributed to the runner. Dependency consistency and bundled model/configuration integrity checks passed. Packaging-tool upgrades succeeded in CI; the local environment's upgrade remained blocked. See `reports/verification.json` and `docs/privacy-security.md` for the exact scope.

## Pending
Main-branch integration and public-visibility approval; Streamlit Community Cloud deployment; actual hosted-browser prediction, upload, download and restart verification. Showing a home page alone will not satisfy deployment acceptance.

The models' discrimination is near chance. Useful operational prioritization has not been established. No live URL or full-completion claim is made at this checkpoint.

# Deploying the evaluation workbench

This is a CPU-only Streamlit application using fixed, actually trained model
artifacts. Near-chance predictive performance does not establish useful clinical
targeting. A public deployment is not an approved environment for confidential
patient information.

## Before deploying

1. Use Python 3.12 in a clean virtual environment. From the repository root,
   upgrade the installer with `python -m pip install --upgrade pip==26.2.1`,
   then install `python -m pip install -r requirements-dev.txt`.
2. Run `python -m pip check`, `python -m pytest -q`, and
   `python scripts/verify_release.py`. Source-dependent tests explicitly skip
   when the private real-source fixture is unavailable. Do not treat those
   skips as real-input verification.
3. For authorized local real-input acceptance, set `CAREACCESS_TEST_DATA` to
   the reviewed adult-source fixture **outside the checkout**, then rerun the
   tests and release verification. Publish aggregate evidence only. Never
   commit source data, fixture rows, per-appointment predictions, secrets or
   patient identifiers. The public CSV template must remain header-only.
4. Review dependency and model integrity results for the exact release commit.
   `models/manifest.json` must cover every fixed artifact and recorded
   configuration hash. Stop if any artifact cannot be verified; do not bypass
   the loader or regenerate evaluation results to make a check pass.

## Streamlit Community Cloud

In your Streamlit Community Cloud workspace, create an app from the reviewed
GitHub repository, choose the release branch (`main` for the existing app), set
the entry point to `app.py`, and select Python 3.12 in advanced settings. The
root `requirements.txt` supplies pinned runtime dependencies and the official
PyTorch CPU-wheel index. This app requires no secrets, paid API or GPU.

Keep `.streamlit/config.toml` in the release: it caps uploads at 5 MB and
disables usage telemetry. Inspect startup logs for installation errors, provider
package replacements and model-load failures. Record the actual Python and
dependency versions; hosting providers can adjust their runtime independently
of the repository. Reboot only when authorized, and verify recovery afterward.

To update the existing deployment, merge the approved, passing release commit
to its configured branch. Verify the provider has loaded that commit before
claiming a fix is live. Consult the provider's current interface/documentation
for account-specific deployment controls; this document does not grant access
or deployment authorization.

## Acceptance and rollback

- Open all seven pages and check model-load failure handling.
- Verify malformed, oversized and unsupported-schema files fail before scoring;
  a header-only file must not generate appointments.
- With an authorized real-source adult fixture, compare hosted single and batch
  outputs for all four models with offline inference within `1e-6`.
- Compare actual downloaded CSV bytes with the displayed single, batch and
  planner results, including review-status edits and formula protection. A
  blocked browser download is an unverified check, not a pass.
- Check planner capacities 0, a limited value and the complete schedule; verify
  stable order and that edits survive capacity changes and page navigation.
- Clear the schedule and withdraw upload authorization; confirm the upload,
  derived results and planner state are removed from the active session. This
  does not erase downloaded copies or provider logs/backups.
- Verify an authorized service restart recovers and rerun inference afterward.

If a release fails, disable its operational use and restore the previously
reviewed release through your normal approved repository/deployment workflow.
Keep model files, manifest and compatible dependencies together. Rerun integrity
and browser checks after rollback. Do not upload replacement model files through
the application or suppress verification errors.

Record passed, failed and unrun checks separately in [release status](release-status.md).

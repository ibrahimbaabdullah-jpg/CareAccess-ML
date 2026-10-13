# Reproduce the bounded predictive-value investigation

These scripts reproduce the declared 2017–2019 temporal development comparison, its once-only historical2020 qualification, aggregate sensitivities, and compute measurements. They never evaluate2021–2022. An authorized local copy of the original Mendeley V1 CSV with SHA256903f30aa008c478dd6451550b53c2bf937653f28763ea288e43f5e99c536cbdf is required. Do not upload it to a service or put it in this repository.

Install these files together under scripts/research. Use the repository's Python environment/dependency versions. Set:

- CAREACCESS_SOURCE: absolute path to the private original CSV outside the repository.
- CAREACCESS_RESEARCH_OUTPUT: a separate private output directory outside the repository and not an ancestor of it.
- CAREACCESS_REPOSITORY: optional absolute checkout root; otherwise inferred from this script's ancestors.
- OMP_NUM_THREADS=2 and OPENBLAS_NUM_THREADS=2 to reproduce the bounded CPU configuration.

The scripts fail closed when required locations are absent or resolve inside the repository, including symlink targets. The source checksum is verified before preparation. All research outputs, including private per-appointment prediction arrays, go to CAREACCESS_RESEARCH_OUTPUT. Never commit that directory or distribute its full contents.

Run in this order from the checkout root:

1. python scripts/research/run_research.py
2. python scripts/research/finish_analysis.py
3. python scripts/research/tie_sensitivity.py
4. python scripts/research/measure_resources.py
5. python scripts/research/validate_research.py

The first script writes the bounded protocol before fits, fixes the learned candidate using development capacity, and evaluates historical2020 only after selection. The second preserves original pooled and2020 tie policies while making annual summaries consistent, adds the two separately predeclared simple calendar baselines, and computes promised AUC/AP uncertainty. The third is a secondary randomized-tie expectation sensitivity and never changes selection. The fourth measures resource-only refits of the same configurations, using actual-source adult appointments for timing. No invented input or synthetic example is used. Timing rows can overlap training and do not constitute predictive validation. The last script verifies temporal boundaries, preserved primary counts, aggregate consistency, resources, and trained artifact integrity.

These are replication scripts, not an invitation to repeatedly optimize against historical2020 or the exposed2021–2022 cohort. An independent untouched dataset is required for a new confirmatory claim. Date-cluster intervals do not account fully for repeated patients, whose identities are unavailable. No raw rows, birth dates, identifiers, outcome reasons, or per-record predictions may be published.

## Provenance

publication_script_hashes.json lists the exact original and publication-adapted SHA256 for every script. Only run_research.py changes location configuration; the other scripts are byte-identical to their corrected original versions. research_paths.py is a new fail-closed location helper. These path adaptations do not change features, model settings, temporal folds, selection, or statistics. The original frozen protocol's code fingerprint identifies the original layout; a new replication naturally records the adapted script's fingerprint.

The original resource protocol initially specified invented timing-only probes. resource_methodology_revised.json documents their replacement with actual-source adult inputs. This amendment changes no predictive fitting/validation experiment or result. Superseded timing measurements and private intermediate artifacts are intentionally excluded from publication.

## Public output allowlist

Aggregate protocol/result JSON, the narrative report, aggregate resource measurements, final verification results, and the independently reviewed conclusions can be published after privacy review. Keep private_development_predictions.npz, all source/fixture CSVs, logs with raw rows, and any unreviewed intermediate outputs private. The scripts themselves never write into models, reports, or any other repository path.

## Independent arithmetic checks

After the bounded experiment, `python scripts/research/review_development_results.py` independently recomputes primary daily-K20 counts and paired intervals from your private output arrays. `python scripts/research/check_frozen_operational_metrics.py` checks the already-published frozen 2021–2022 cohort without fitting or selection. This last check is explicitly historical verification, not part of the <2021 experiment or a new independent evaluation. Both write aggregates only to the configured external output directory. These two additional review scripts have path-only publication adaptations; their original and published hashes are recorded in publication provenance.

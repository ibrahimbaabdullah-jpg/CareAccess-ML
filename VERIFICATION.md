# Static verification for the frozen browser research demonstration

Run from this directory with the installed Python 3 and Node.js runtimes:

```sh
python3 scripts/verify_static.py
```

No packages are downloaded or installed. No credentials, source records,
patient/appointment keys, example inputs, outcome labels, or private test
fixtures are needed or included.

## What this check establishes

- `site-manifest.json` lists only the eleven original site files, their byte
  lengths, and SHA-256 checksums. The verifier checks all eleven byte-for-byte,
  rejects duplicate JSON keys and unexpected paths, and rejects unexpected
  files or symlinks in the release tree (Git checkout metadata is excluded).
- The model has the expected frozen identity, objective, numeric/categorical
  contract, 108 transformed features, and 300 trees with 3,692 total nodes.
  Tree arrays, child links, reachability, depth, feature indices, finite
  thresholds/leaf weights, and preprocessing dimensions are checked.
- The application pins the frozen model checksum and size before using it.
- Node checks the syntax of all three JavaScript modules and calls the existing
  `validateModel` function on the real frozen JSON. No input record is scored.

The manifest is an integrity record, not a digital signature or independent
proof of authenticity. Changes to code, verification logic, manifests, or the
model still require review. Passing this check does not authorize publication.

## What it does not establish

Positive numerical parity was evaluated privately and offline on 15,016 actual,
target-free source input records. Public CI does not rerun that evidence and
contains neither those records nor generated positive synthetic records.
The previously reported transform/margin equality and small probability
rounding differences are separate evidence, not results of this static job.

This job does not exercise browser behavior, responsive layout, accessibility,
network transmission, browser storage, reset/lifecycle interactions, or clinical
performance. It does not certify privacy, anonymization, fairness, or clinical
readiness. Those require separate checks appropriate to their scope.

## CI and deployment boundaries

The workflow runs on pushes to and pull requests targeting
`research-browser-demo`. Its job/check name is `static-check`. It uses only
`contents: read`, disables persisted checkout credentials, and pins the official
checkout action to a full commit SHA. The pin matches the existing main-branch
workflow and the official `actions/checkout` v4 reference checked on
13 October 2026. Runner-provided Python and Node are used without additional
actions or dependencies.

This workflow does not deploy, publish artifacts, upload files, modify Pages,
create credentials, or change branch protections. GitHub Pages built-in
publication remains a separate configuration and authorization decision.

Independent application code retains its MIT license. The model and fitted
preprocessing asset retain their separate CC BY-NC-SA 4.0 notice; the code
license does not relicense those assets. See the adjacent license files and
the notice beside the downloadable model.

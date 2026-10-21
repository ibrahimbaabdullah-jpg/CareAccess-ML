# Local testing on any platform

Two things stop the suite from running on a fresh machine: the pinned PyTorch build is not
published for macOS, and the source-dependent tests need a fixture of actual rows that is
deliberately absent from this repository. This page covers both.

## Why `pip install -r requirements-dev.txt` fails on macOS

`requirements.txt` pins `torch==2.14.1+cpu`. The `+cpu` local-version suffix is published only
for Linux and Windows, where a CPU build must be distinguished from a CUDA build; macOS wheels
carry no such suffix, so the pin cannot resolve there. Because pip resolves the file as one unit,
nothing installs at all and the missing `pytest` or `numpy` that follows is a symptom, not the
cause.

On Apple Silicon with macOS 14 or newer the same version exists as
`torch-2.14.1-cp312-cp312-macosx_14_0_arm64.whl`, so installing from a copy of
`requirements.txt` with the suffix removed works, at the cost of a different platform build than
CI uses. On an Intel Mac no build of this version exists at all (the last is 2.2.2), so use
Docker.

## Docker

```
docker build -t careaccess-test .
docker run --rm careaccess-test
```

That runs the public suite exactly as CI does. The image installs dependencies before copying
the rest of the tree, so editing code and rerunning does not reinstall PyTorch. To run against
your working copy without rebuilding, mount it:

```
docker run --rm -v "$PWD:/app" careaccess-test
```

## The source-dependent fixture

Roughly 53 tests skip unless `CAREACCESS_TEST_DATA` points at a CSV of actual appointment rows
in the public template layout. The file is never committed: `scripts/verify_release.py` fails the
release if any row-level CSV appears in the tree, and the template in `examples/` must stay
header-only. Build it outside the checkout instead.

1. Download *Medical Appointments No-Show*, Mendeley Data version 1
   ([DOI 10.17632/wm6w2fvkfj.1](https://data.mendeley.com/datasets/wm6w2fvkfj/1), CC BY 4.0).
   `careaccess/data.py` verifies its SHA-256, so a different version or an edited copy is
   rejected with a checksum error rather than silently accepted.
2. Build the fixture:

```
python scripts/make_test_fixture.py --source ~/mendeley-source.csv --out ~/careaccess-private/fixture.csv
```

The script refuses to write inside the repository. It selects 24 adult appointments dated
outside the model's training window, assigns fresh non-identifying references, keeps only the
four template columns, and drops the outcome and demographic columns. It then checks every row
against the input schema and confirms each model returns finite, varying probabilities, so a
fixture that would fail the suite is rejected at build time rather than during testing.

3. Run the full suite:

```
export CAREACCESS_TEST_DATA=~/careaccess-private/fixture.csv
python -m pytest -q
python scripts/verify_release.py
```

Export the variable rather than prefixing one command, so the release check sees the fixture too.

On a machine with no local install, such as an Intel Mac, both steps run in the container: the
fixture builder loads the models, so it needs the same dependencies the suite does. Keep the
source and the fixture in a directory of their own, mounted outside `/app`, so neither can land
in the repository.

```
mkdir -p ~/careaccess-private
docker run --rm -v "$PWD:/app" -v ~/careaccess-private:/out \
  -v ~/Downloads/mendeley-source.csv:/src/source.csv:ro careaccess-test \
  python scripts/make_test_fixture.py --source /src/source.csv --out /out/fixture.csv
```

Then run the suite against it:

```
docker run --rm -v "$PWD:/app" -v ~/careaccess-private/fixture.csv:/data/fixture.csv:ro \
  -e CAREACCESS_TEST_DATA=/data/fixture.csv careaccess-test
```

A passing run reports no failures and no skips from the source-dependent suites, and
`verify_release.py` stops printing `Actual-source inference parity NOT RUN`. A provenance note
about changed training sources is expected and is not a failure.

## What these tests do and do not show

They exercise parsing, validation, inference plumbing and the application on real rows rather
than generated ones. They are software checks. A fixture rebuilt this way is not the file behind
the verification records in `docs/release-status.md`; the counts and probabilities recorded there
refer to the earlier fixture, so a rebuilt run reproduces the contract, not those exact numbers. The rows come from the published source and sit
inside the frozen evaluation period, so they are not new evidence: they say nothing about
predictive performance or clinical benefit, which `docs/validation-protocol.md` addresses.

Never commit the fixture, attach it to an issue or pull request, upload it to a hosted service,
or paste rows into a chat. Keep it on the machine that runs the tests.

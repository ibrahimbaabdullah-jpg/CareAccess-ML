"""Write the private actual-source test fixture outside the repository. Run from the repository root."""

import argparse
from pathlib import Path

import numpy as np

from careaccess.data import prepare_source
from careaccess.fixture import select_fixture_rows
from careaccess.inference import load_bundle, model_names, predict

ROOT = Path(__file__).resolve().parents[1]


def main(args):
    out = Path(args.out).expanduser().resolve()
    if out == ROOT or ROOT in out.parents:
        raise SystemExit(f"Refusing to write patient rows inside the repository: {out}")
    frame, _ = prepare_source(args.source)
    bundle = load_bundle()
    fixture = select_fixture_rows(frame, bundle["feature_contract"])
    for name in model_names(bundle):
        probabilities = predict(fixture, name, bundle)
        if not np.isfinite(probabilities).all():
            raise SystemExit(f"{name} returned non-finite probabilities for the selected rows.")
        if name != "Constant baseline" and np.ptp(probabilities) <= 1e-5:
            raise SystemExit(f"{name} scores the selected rows almost identically; the tests need varied inputs.")
    out.parent.mkdir(parents=True, exist_ok=True)
    fixture.to_csv(out, index=False)
    print(f"Wrote {len(fixture)} rows to {out}")
    print("Keep this file out of the repository and off any shared service.")
    print(f"Export it for the tests and the release check: export CAREACCESS_TEST_DATA={out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Verified Mendeley version 1 source CSV")
    parser.add_argument("--out", required=True, help="Destination outside the repository")
    main(parser.parse_args())

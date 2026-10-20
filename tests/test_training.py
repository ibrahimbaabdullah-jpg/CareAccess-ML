"""End-to-end training mechanics on generated rows, written to a temporary folder."""

import hashlib
import json

import numpy as np
import pytest

from careaccess import data, inference, training


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    from conftest import make_source

    root = tmp_path_factory.mktemp("trained")
    source = root / "generated_source.csv"
    make_source().to_csv(source, index=False)
    patcher = pytest.MonkeyPatch()
    patcher.setattr(data, "SOURCE_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    try:
        default, report = training.train(source, root=root)
    finally:
        patcher.undo()
    return root, default, report


def test_training_reports_every_model_without_using_test_for_selection(trained):
    _, default, report = trained
    names = [row["model"] for row in report["metrics"]]
    assert names == list(training.SLUGS)
    assert default != training.BASELINE
    assert report["selection"]["default_model"] == default
    assert {row["model"] for row in report["selection"]["validation"]} == set(names)
    assert set(report["curves"]) == set(report["capacity"]) == set(report["uncertainty"]) == set(names)


def test_retrained_artifacts_load_through_the_safe_loader(trained, monkeypatch, generated_schedule):
    root, default, _ = trained
    manifest = json.loads((root / "models/manifest.json").read_text())
    assert manifest["skops_types"] == inference.SKOPS_TYPES
    assert manifest["model_files"] == inference.MODEL_FILES
    assert set(manifest["configuration_sha256"]) == set(training.CONFIGURATION_FILES)
    monkeypatch.setattr(inference, "ROOT", root)
    bundle = inference.load_bundle()
    assert bundle["default_model"] == default
    for name in inference.model_names(bundle):
        scores = inference.predict(generated_schedule, name, bundle)
        assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()

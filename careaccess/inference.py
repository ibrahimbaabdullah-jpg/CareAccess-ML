"""Real CPU inference from checked, bundled trained artifacts only."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import skops.io as sio
import torch
from safetensors.torch import load_file

from careaccess.calibration import apply_calibration
from careaccess.network import TabularMLP
from careaccess.validation import make_features, validate_frame

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILES = {
    "Constant baseline": "baseline.skops",
    "Logistic Regression": "logistic.skops",
    "Gradient Boosting": "gradient.skops",
    "PyTorch Neural Network": "neural.safetensors",
}
# The audited deserialization allowlist is code-owned. A manifest cannot expand it.
SKOPS_TYPES = {
    "preprocessor.skops": ["numpy.dtype"],
    "baseline.skops": [],
    "logistic.skops": [],
    "gradient.skops": ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"],
}


def verify_artifacts(folder, manifest):
    """Reject incomplete or redirected manifests before deserializing anything."""
    if manifest.get("model_files") != MODEL_FILES:
        raise ValueError("Unexpected bundled model files.")
    required = {"preprocessor.skops", *MODEL_FILES.values()}
    checksums = manifest.get("sha256")
    if not isinstance(checksums, dict) or set(checksums) != required:
        raise ValueError("Every required model artifact must have exactly one checksum.")
    if manifest.get("skops_types") != SKOPS_TYPES:
        raise ValueError("Unexpected model type allowlist.")
    for name, digest in checksums.items():
        path = folder / name
        if Path(name).name != name or path.resolve().parent != folder.resolve():
            raise ValueError("Invalid bundled artifact name or location.")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("Invalid model checksum.")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Model checksum mismatch; deployment must be repaired.")


def model_names(bundle=None):
    bundle = load_bundle() if bundle is None else bundle
    return list(bundle["manifest"]["model_files"])


def scoring_model_names(bundle=None):
    """Models offered for interactive scoring.

    A negative calibration slope reverses the raw score ordering, so that model
    stays visible in evaluation pages but is not offered for ranking inputs.
    """
    bundle = load_bundle() if bundle is None else bundle
    calibrators = bundle["manifest"]["calibrators"]
    return [name for name in model_names(bundle) if not (calibrators.get(name) or {}).get("coef", 0) < 0]


def load_bundle():
    folder = ROOT / "models"
    manifest = json.loads((folder / "manifest.json").read_text())
    models = {}
    verify_artifacts(folder, manifest)

    def read_skops(name):
        unknown = sio.get_untrusted_types(file=folder / name)
        if unknown != SKOPS_TYPES[name]:
            raise ValueError("Unexpected model types.")
        return sio.load(folder / name, trusted=SKOPS_TYPES[name])

    pre = read_skops("preprocessor.skops")
    for label, name in manifest["model_files"].items():
        if name.endswith(".safetensors"):
            model = TabularMLP(manifest["input_dim"])
            model.load_state_dict(load_file(str(folder / name), device="cpu"))
            model.eval()
        else:
            model = read_skops(name)
        models[label] = model
    torch.set_num_threads(2)
    return {
        "manifest": manifest,
        "models": models,
        "preprocessor": pre,
        "default_model": manifest["default_model"],
        "feature_contract": manifest["feature_contract"],
        "report": json.loads((ROOT / "reports/evaluation.json").read_text()),
    }


def predict(frame, model_name=None, bundle=None):
    bundle = load_bundle() if bundle is None else bundle
    clean, errors = validate_frame(frame)
    if len(errors):
        raise ValueError("Invalid appointment input: " + "; ".join(errors.message.unique()))
    name = model_name or bundle["default_model"]
    if name not in bundle["models"]:
        raise ValueError("Unknown model.")
    if len(clean) == 0:
        return np.array([], dtype=float)
    x = bundle["preprocessor"].transform(make_features(clean)).astype("float32")
    model = bundle["models"][name]
    if isinstance(model, TabularMLP):
        with torch.no_grad():
            p = torch.sigmoid(model(torch.tensor(x))).numpy().astype(float)
    else:
        p = model.predict_proba(x)[:, 1]
    p = apply_calibration(p, bundle["manifest"]["calibrators"][name])
    if not np.isfinite(p).all():
        raise ValueError("Non-finite model result.")
    return np.asarray(p, dtype=float)

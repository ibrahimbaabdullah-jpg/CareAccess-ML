"""Chronological training, calibration and frozen evaluation of the benchmark models.

The steps run in a fixed order so seeded results stay reproducible: the
scikit-learn candidates are fitted before the network consumes the torch
random stream. Model selection, calibration and thresholds use only the 2019
validation and 2020 calibration periods; the final test is scored afterwards.
"""

import copy
import hashlib
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import skops.io as sio
import torch
from safetensors.torch import save_file
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from careaccess.calibration import apply_calibration, logit
from careaccess.data import SOURCE_SHA256, prepare_source, temporal_split
from careaccess.evaluation import bootstrap_intervals, capacity_metrics, curves, evaluate
from careaccess.network import TabularMLP
from careaccess.validation import INPUT_COLUMNS, SPECIALTIES, make_features

SEED = 20261013
ROOT = Path(__file__).resolve().parents[1]
BASELINE = "Constant baseline"
NETWORK = "PyTorch Neural Network"
SLUGS = {
    BASELINE: "baseline",
    "Logistic Regression": "logistic",
    "Gradient Boosting": "gradient",
    NETWORK: "neural",
}
CONFIGURATION_FILES = [
    "scripts/train.py",
    "careaccess/training.py",
    "careaccess/calibration.py",
    "careaccess/network.py",
    "careaccess/data.py",
    "careaccess/validation.py",
    "scripts/feature_ablation.py",
    "careaccess/evaluation.py",
]
SOURCE = {
    "title": "Medical Appointments No-Show",
    "author": "Luiz Henrique Salazar",
    "doi": "10.17632/wm6w2fvkfj.1",
    "version": 1,
    "url": "https://data.mendeley.com/datasets/wm6w2fvkfj/1",
    "license": "CC BY 4.0",
}
SELECTION_RULE = (
    "Highest 2019 validation average precision among the three trained candidates; "
    "tie break log loss, then name. No final-test results used."
)
LIMITATIONS = [
    "Single historical Brazilian rehabilitation service; not locally validated elsewhere.",
    "No patient identifiers: repeated persons may span periods; unseen-patient generalization and "
    "patient-cluster intervals cannot be established.",
    "Exact full-row duplicates removed; some identical distinct encounters may have been collapsed.",
    "Age recomputed from birth date; provided raw age and all age flags excluded. Birth dates are never published.",
    "Appointment snapshot availability is assumed, not proven by versioned booking records.",
    "Outcome prevalence and service activity changed over time, including the pandemic.",
    "Calibration uses only 827 appointments and 117 missed visits from sparse 2020 data.",
    "Date-cluster confidence intervals do not account fully for repeated patients.",
    "Predicted missed attendance is not predicted benefit from outreach. No causal or prospective reduction "
    "in no-shows is established.",
    "Public application requires authorized nonidentifying inputs; identifiable health information must not be uploaded.",
    "Probabilities outside historical support or in a new institution may be unreliable.",
    "Daily top-K evaluation counts potential missed visits identified, not prevented.",
]
AUDIT_ORDER = (
    "An initial automated fixed-config run generated test metrics before the additional-input clarification. "
    "Subsequent disability/city sensitivity analysis used only training/2019 validation and did not change the "
    "original features or selection rule. Final-test results were not used for feature, model, calibration or "
    "threshold selection. A later verification run repeated the unchanged fit and corrected only "
    "label-independent ties in capacity evaluation."
)
FEATURE_CONTRACT_DESCRIPTION = (
    "Age at appointment, specialty and scheduled calendar/time features only. Blank age imputed using training "
    "median with missingness indicator; blank specialty gets a missing category."
)
FIT_PERIODS = ["validation", "calibration"]


def seed_everything():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)


def build_preprocessor():
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())])
    return ColumnTransformer(
        [
            ("numeric", numeric, ["age", "hour", "weekday", "month"]),
            ("specialty", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["specialty"]),
        ]
    )


def candidate_models():
    """All hyperparameters are fixed before training."""
    return {
        BASELINE: DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(C=1, max_iter=1000, random_state=SEED),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=150,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            l2_regularization=2,
            early_stopping=False,
            random_state=SEED,
        ),
    }


def transform_splits(preprocessor, features):
    """Fit imputation, scaling and encoding on training rows only."""
    matrices = {"train": preprocessor.fit_transform(features["train"]).astype("float32")}
    matrices.update({k: preprocessor.transform(features[k]).astype("float32") for k in ["validation", "calibration", "test"]})
    return matrices


def fit_candidates(candidates, z, y):
    raw_scores, fit_times = {}, {}
    for name, model in candidates.items():
        started = time.monotonic()
        model.fit(z["train"], y["train"])
        fit_times[name] = time.monotonic() - started
        raw_scores[name] = {k: model.predict_proba(z[k])[:, 1] for k in FIT_PERIODS}
    return raw_scores, fit_times


def network_scores(net, matrix):
    with torch.no_grad():
        return torch.sigmoid(net(torch.tensor(matrix))).numpy()


def train_network(z, y, max_epochs=40, patience_limit=6, batch_size=256):
    """AdamW training with early stopping on 2019 validation log loss."""
    net = TabularMLP(z["train"].shape[1])
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.001)
    tx = torch.tensor(z["train"])
    ty = torch.tensor(y["train"], dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best, best_state, patience, history = float("inf"), None, 0, []
    started = time.monotonic()
    for epoch in range(1, max_epochs + 1):
        net.train()
        losses = []
        for ix in torch.randperm(len(tx)).split(batch_size):
            optimizer.zero_grad()
            loss = loss_fn(net(tx[ix]), ty[ix])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        net.eval()
        validation_loss = float(log_loss(y["validation"], network_scores(net, z["validation"])))
        history.append({"epoch": epoch, "train_bce": float(np.mean(losses)), "validation_log_loss": validation_loss})
        if validation_loss < best - 1e-5:
            best, best_state, patience = validation_loss, copy.deepcopy(net.state_dict()), 0
        else:
            patience += 1
        if patience >= patience_limit:
            break
    net.load_state_dict(best_state)
    net.eval()
    return net, history, time.monotonic() - started


def select_default(raw_scores, y):
    """Default chosen solely by validation AP; no final-test selection or refitting."""
    validation = [
        {
            "model": name,
            "average_precision": float(average_precision_score(y["validation"], p["validation"])),
            "log_loss": float(log_loss(y["validation"], p["validation"])),
        }
        for name, p in raw_scores.items()
    ]
    eligible = [v for v in validation if v["model"] != BASELINE]
    default = sorted(eligible, key=lambda v: (-v["average_precision"], v["log_loss"], v["model"]))[0]["model"]
    return default, validation


def calibrate(raw_scores, y):
    """Fit sigmoid calibrators and F1 thresholds on the 2020 calibration period."""
    calibrators, thresholds = {}, {}
    grid = np.linspace(0.05, 0.95, 91)
    for name, p in raw_scores.items():
        if name == BASELINE:
            calibrators[name] = None
            calibrated = p["calibration"]
        else:
            features = logit(p["calibration"]).reshape(-1, 1)
            model = LogisticRegression(C=1, max_iter=1000, random_state=SEED).fit(features, y["calibration"])
            calibrators[name] = {"coef": float(model.coef_[0, 0]), "intercept": float(model.intercept_[0])}
            calibrated = model.predict_proba(features)[:, 1]
        thresholds[name] = float(
            max(grid, key=lambda t, c=calibrated: f1_score(y["calibration"], c >= t, zero_division=0))
        )
    return calibrators, thresholds


def save_models(models_dir, preprocessor, candidates, net):
    models_dir.mkdir(exist_ok=True)
    sio.dump(preprocessor, models_dir / "preprocessor.skops")
    files = {}
    for name, model in candidates.items():
        files[name] = SLUGS[name] + ".skops"
        sio.dump(model, models_dir / files[name])
    files[NETWORK] = SLUGS[NETWORK] + ".safetensors"
    save_file(net.state_dict(), str(models_dir / files[NETWORK]))
    return files


def age_band(ages):
    bands = pd.cut(ages, [-0.01, 18, 60, 111], right=False, labels=["Under 18", "18–59", "60+"])
    return bands.astype(object).fillna("Unknown")


def subgroup_rows(name, test, y, p, threshold):
    """Descriptive subgroup metrics; groups under 100 rows or 20 events/non-events are suppressed."""
    rows = []
    for dimension, groups in [("gender", test.gender_audit), ("age", age_band(test.age))]:
        for group in pd.unique(groups):
            mask = np.asarray(groups == group)
            gy = y[mask]
            if mask.sum() < 100 or gy.sum() < 20 or (len(gy) - gy.sum()) < 20:
                continue
            rows.append({
                "model": name,
                "dimension": dimension,
                "group": str(group),
                "rows": int(mask.sum()),
                "missed": int(gy.sum()),
                **evaluate(gy, p[mask], threshold),
            })
    return rows


def evaluate_final_test(report, candidates, net, z, y, test, calibrators, thresholds):
    """Score the held-out test period once, after every selection decision is fixed."""
    for name in calibrators:
        p = network_scores(net, z["test"]).astype(float) if name == NETWORK else candidates[name].predict_proba(z["test"])[:, 1]
        p = apply_calibration(p, calibrators[name])
        report["metrics"].append({"model": name, **evaluate(y["test"], p, thresholds[name])})
        report["curves"][name] = curves(y["test"], p)
        report["capacity"][name] = capacity_metrics(y["test"], p, test.appointment_datetime)
        report["uncertainty"][name] = bootstrap_intervals(y["test"], p, test.appointment_datetime)
        report["subgroups"].extend(subgroup_rows(name, test, y["test"], p, thresholds[name]))


def feature_importance(model, preprocessor, z, y):
    """Validation-only permutation importance: descriptive association, not causal explanation."""
    result = permutation_importance(
        model, z["validation"], y["validation"], scoring="average_precision", n_repeats=5, random_state=SEED
    )
    return [
        {"feature": str(n), "mean_ap_decrease": float(m), "std": float(s)}
        for n, m, s in zip(preprocessor.get_feature_names_out(), result.importances_mean, result.importances_std, strict=True)
    ]


def build_manifest(root, models_dir, default, files, calibrators, thresholds, input_dim, versions, frame, train):
    manifest = {
        "version": "0.1.0",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "default_model": default,
        "model_files": files,
        "calibrators": calibrators,
        "thresholds": thresholds,
        "input_dim": input_dim,
        "versions": versions,
        "sha256": {},
        "skops_types": {},
    }
    for path in sorted(models_dir.iterdir()):
        if path.suffix in [".skops", ".safetensors"]:
            manifest["sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            if path.suffix == ".skops":
                manifest["skops_types"][path.name] = sio.get_untrusted_types(file=path)
    manifest["feature_contract"] = {
        "input_columns": INPUT_COLUMNS,
        "specialty_values": SPECIALTIES,
        "age_min": 0,
        "age_max": 110,
        "age_optional": True,
        "specialty_optional": True,
        "datetime_format": "YYYY-MM-DDTHH:MM",
        "train_start": train.appointment_datetime.min(),
        "train_end": train.appointment_datetime.max(),
        "data_start": frame.appointment_datetime.min(),
        "data_end": frame.appointment_datetime.max(),
        "description": FEATURE_CONTRACT_DESCRIPTION,
    }
    manifest["configuration_sha256"] = {
        f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in CONFIGURATION_FILES
    }
    return manifest


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def train(raw, root=ROOT):
    """Train, calibrate, evaluate and save every artifact under ``root``."""
    started = time.monotonic()
    seed_everything()
    frame, counts = prepare_source(raw)
    parts = temporal_split(frame)
    features = {k: make_features(v) for k, v in parts.items()}
    y = {k: v.target.to_numpy(dtype=int) for k, v in parts.items()}
    preprocessor = build_preprocessor()
    z = transform_splits(preprocessor, features)

    candidates = candidate_models()
    raw_scores, fit_times = fit_candidates(candidates, z, y)
    net, history, fit_times[NETWORK] = train_network(z, y)
    raw_scores[NETWORK] = {k: network_scores(net, z[k]).astype(float) for k in FIT_PERIODS}

    default, validation = select_default(raw_scores, y)
    calibrators, thresholds = calibrate(raw_scores, y)
    models_dir = root / "models"
    files = save_models(models_dir, preprocessor, candidates, net)

    report = {
        "source": {**SOURCE, "sha256": SOURCE_SHA256},
        "counts": counts,
        "splits": {
            k: {
                "rows": len(v),
                "missed": int(v.target.sum()),
                "start": v.appointment_datetime.min(),
                "end": v.appointment_datetime.max(),
            }
            for k, v in parts.items()
        },
        "selection": {"default_model": default, "rule": SELECTION_RULE, "validation": validation},
        "metrics": [],
        "curves": {},
        "capacity": {},
        "subgroups": [],
        "uncertainty": {},
        "fit_seconds": fit_times,
        "neural_history": history,
        "limitations": LIMITATIONS,
    }
    evaluate_final_test(report, candidates, net, z, y, parts["test"], calibrators, thresholds)
    report["feature_importance"] = (
        feature_importance(candidates[default], preprocessor, z, y) if default in candidates else []
    )

    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
    }
    manifest = build_manifest(
        root, models_dir, default, files, calibrators, thresholds, int(z["train"].shape[1]), versions, frame, parts["train"]
    )
    report.update(
        training_seconds=time.monotonic() - started,
        versions=versions,
        created_utc=manifest["created_utc"],
        default_model=default,
        audit_order=AUDIT_ORDER,
    )
    ablation = root / "reports/feature_ablation.json"
    if ablation.exists():
        report["feature_ablation"] = json.loads(ablation.read_text())
    save_json(models_dir / "manifest.json", manifest)
    save_json(root / "reports/evaluation.json", report)
    return default, report

"""Evaluation on held-out dates, including same-day operational capacity."""

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = p >= threshold
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def curves(y: np.ndarray, p: np.ndarray) -> dict:
    fpr, tpr, _ = roc_curve(y, p)
    precision, recall, _ = precision_recall_curve(y, p)
    obs, est = calibration_curve(y, p, n_bins=10, strategy="quantile")

    def thin(a, b, n=150):
        ind = np.unique(np.linspace(0, len(a) - 1, min(n, len(a))).astype(int))
        return np.asarray(a)[ind].tolist(), np.asarray(b)[ind].tolist()

    fpr, tpr = thin(fpr, tpr)
    precision, recall = thin(precision, recall)
    return {
        "roc": {"fpr": fpr, "tpr": tpr},
        "pr": {"precision": precision, "recall": recall},
        "calibration": {"predicted": est.tolist(), "observed": obs.tolist()},
    }


def capacity_metrics(y: np.ndarray, p: np.ndarray, dates: pd.Series) -> list[dict]:
    data = pd.DataFrame({"y": np.asarray(y), "p": p, "date": pd.to_datetime(dates).dt.date.to_numpy()})
    data["tie"] = np.random.default_rng(20261013).random(len(data))
    rows = []
    for k in [5, 10, 20]:
        selected = (
            data.sort_values(["p", "tie"], ascending=[False, True], kind="stable").groupby("date", sort=False).head(k)
        )
        hits = int(selected.y.sum())
        n = len(selected)
        total = int(data.y.sum())
        expected = sum(min(k, len(g)) * g.y.mean() for _, g in data.groupby("date"))
        rows.append(
            {
                "k": k,
                "selected": n,
                "captured_missed": hits,
                "precision_at_k": hits / n if n else 0,
                "recall_at_k": hits / total if total else 0,
                "random_expected_precision_at_k": expected / n if n else 0,
                "random_expected_recall_at_k": expected / total if total else 0,
                "scope": "Up to K per appointment day; seeded label-independent random tie breaking",
            }
        )
    return rows


def bootstrap_intervals(
    y: np.ndarray, p: np.ndarray, dates: pd.Series, seed: int = 20261013, n_boot: int = 200
) -> dict:
    """Date-cluster bootstrap; patient clusters unavailable. Fixed frozen predictions."""
    y = np.asarray(y)
    p = np.asarray(p)
    days = pd.to_datetime(dates).dt.date.to_numpy()
    unique = np.unique(days)
    groups = [np.flatnonzero(days == d) for d in unique]
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ["roc_auc", "average_precision", "brier"]}
    for _ in range(n_boot):
        ix = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if len(np.unique(y[ix])) < 2:
            continue
        out["roc_auc"].append(roc_auc_score(y[ix], p[ix]))
        out["average_precision"].append(average_precision_score(y[ix], p[ix]))
        out["brier"].append(brier_score_loss(y[ix], p[ix]))
    return {
        k: {"lower": float(np.percentile(v, 2.5)), "upper": float(np.percentile(v, 97.5)), "replicates": len(v)}
        for k, v in out.items()
    }

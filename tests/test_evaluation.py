"""Evaluation helpers on small hand-checkable inputs."""

import numpy as np
import pandas as pd
import pytest

from careaccess.evaluation import bootstrap_intervals, capacity_metrics, curves, evaluate


def test_evaluate_perfect_ranking_and_confusion_matrix():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.7])
    result = evaluate(y, p, threshold=0.5)
    assert result["roc_auc"] == 1.0
    assert result["average_precision"] == 1.0
    assert result["precision"] == result["recall"] == result["f1"] == 1.0
    assert result["confusion_matrix"] == [[3, 0], [0, 3]]
    assert result["brier"] == pytest.approx(np.mean((p - y) ** 2))


def test_evaluate_threshold_above_every_score_predicts_no_misses():
    result = evaluate(np.array([0, 1, 0, 1]), np.array([0.1, 0.2, 0.3, 0.4]), threshold=0.95)
    assert result["precision"] == result["recall"] == 0.0
    assert result["confusion_matrix"] == [[2, 0], [2, 0]]


def test_curves_are_thinned_and_cover_both_ends():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = rng.random(2000)
    result = curves(y, p)
    assert len(result["roc"]["fpr"]) <= 150
    assert result["roc"]["fpr"][0] == 0.0 and result["roc"]["fpr"][-1] == 1.0
    assert len(result["pr"]["precision"]) == len(result["pr"]["recall"]) <= 150
    assert len(result["calibration"]["predicted"]) == len(result["calibration"]["observed"]) <= 10


def test_capacity_metrics_select_up_to_k_per_day():
    dates = pd.Series(["2021-01-04T09:00"] * 8 + ["2021-01-05T09:00"] * 3)
    y = np.array([1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.5, 0.4, 0.3])
    rows = {row["k"]: row for row in capacity_metrics(y, p, dates)}
    assert rows[5]["selected"] == 5 + 3
    assert rows[10]["selected"] == rows[20]["selected"] == 11
    # Day one's top five are 0.9, 0.8, 0.2 and two of the tied 0.1 scores.
    assert rows[5]["captured_missed"] == 1 + 1 + 1
    assert rows[5]["random_expected_precision_at_k"] == pytest.approx((5 * 2 / 8 + 3 * 1 / 3) / 8)
    assert rows[20]["recall_at_k"] == 1.0


def test_bootstrap_intervals_are_ordered_and_reproducible():
    rng = np.random.default_rng(1)
    dates = pd.Series(pd.date_range("2021-01-01", periods=30).repeat(20).strftime("%Y-%m-%dT%H:%M"))
    y = rng.integers(0, 2, len(dates))
    p = np.clip(y * 0.3 + rng.random(len(dates)) * 0.7, 0, 1)
    first = bootstrap_intervals(y, p, dates, n_boot=50)
    assert first == bootstrap_intervals(y, p, dates, n_boot=50)
    for metric in ["roc_auc", "average_precision", "brier"]:
        assert first[metric]["lower"] <= first[metric]["upper"]
        assert first[metric]["replicates"] == 50

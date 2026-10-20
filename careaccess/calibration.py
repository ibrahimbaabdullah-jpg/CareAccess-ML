"""Sigmoid (Platt) recalibration shared by training and inference."""

import numpy as np

EPSILON = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPSILON, 1 - EPSILON)
    return np.log(p / (1 - p))


def apply_calibration(p: np.ndarray, calibrator: dict | None) -> np.ndarray:
    """Map raw probabilities through a fitted sigmoid; ``None`` leaves them unchanged."""
    if not calibrator:
        return p
    return 1 / (1 + np.exp(-(calibrator["coef"] * logit(p) + calibrator["intercept"])))

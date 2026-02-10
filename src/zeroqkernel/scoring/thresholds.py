"""Threshold calibration utilities."""

from __future__ import annotations

import numpy as np


def calibrate_quantile(scores: np.ndarray, false_positive_rate: float = 0.01) -> float:
    """Set threshold at upper quantile of normal scores."""
    q = 1.0 - false_positive_rate
    return float(np.quantile(scores, q))


def apply_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Return binary decisions where 1 indicates anomaly."""
    return (scores >= threshold).astype(int)

"""Anomaly score builders for kernel-based IDS."""

from __future__ import annotations

import numpy as np


def mean_kernel_similarity(k_eval_train: np.ndarray) -> np.ndarray:
    """Higher value means more similar to training normals."""
    return k_eval_train.mean(axis=1)


def mmd_score(k_eval_train: np.ndarray) -> np.ndarray:
    """Placeholder proxy; replace with full MMD formulation."""
    sim = mean_kernel_similarity(k_eval_train)
    return 1.0 - sim

"""Classical RBF and Laplacian kernel utilities."""

from __future__ import annotations

import numpy as np


def rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
    x2 = np.sum(x * x, axis=1, keepdims=True)
    y2 = np.sum(y * y, axis=1, keepdims=True).T
    dist2 = np.maximum(x2 + y2 - 2.0 * x @ y.T, 0.0)
    return np.exp(-gamma * dist2)


def laplacian_kernel(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
    dists = np.abs(x[:, None, :] - y[None, :, :]).sum(axis=2)
    return np.exp(-gamma * dists)

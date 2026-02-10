"""Feature engineering utilities."""

from __future__ import annotations

import numpy as np


def zscore_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return (x - mean) / (std + eps)


def minmax_scale(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    low = x.min(axis=0, keepdims=True)
    high = x.max(axis=0, keepdims=True)
    return (x - low) / (high - low + eps)


def window_sequence(x: np.ndarray, window_size: int) -> np.ndarray:
    """Create fixed windows for sequential traffic data.

    TODO:
    - choose stride and padding policy from config
    """
    if window_size <= 1:
        return x
    raise NotImplementedError("Windowing skeleton only")

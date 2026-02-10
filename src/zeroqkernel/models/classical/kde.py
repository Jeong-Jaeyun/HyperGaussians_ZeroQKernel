"""Kernel density estimation baseline wrapper."""

from __future__ import annotations


class KDEBaseline:
    """Skeleton for density-based anomaly baseline."""

    def __init__(self, bandwidth: float = 1.0) -> None:
        self.bandwidth = bandwidth

    def fit(self, x_train):
        _ = x_train
        raise NotImplementedError("KDE baseline not implemented in scaffold")

    def score_samples(self, x_eval):
        _ = x_eval
        raise NotImplementedError("KDE baseline not implemented in scaffold")

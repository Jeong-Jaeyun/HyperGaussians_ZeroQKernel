"""One-class anomaly model with quantum kernels."""

from __future__ import annotations


class QuantumOneClassModel:
    """Skeleton for one-class decision function over quantum kernels."""

    def __init__(self, nu: float = 0.05) -> None:
        self.nu = nu

    def fit(self, k_train):
        _ = k_train
        raise NotImplementedError("Quantum one-class fit not implemented")

    def score_samples(self, k_eval):
        _ = k_eval
        raise NotImplementedError("Quantum one-class scoring not implemented")

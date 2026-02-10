"""One-class SVM baseline wrapper."""

from __future__ import annotations


class OCSVMBaseline:
    """Skeleton wrapper for one-class SVM with precomputed kernels."""

    def __init__(self, nu: float = 0.05) -> None:
        self.nu = nu
        self._model = None

    def fit(self, k_train):
        _ = k_train
        raise NotImplementedError("OCSVM training not implemented in scaffold")

    def score_samples(self, k_eval):
        _ = k_eval
        raise NotImplementedError("OCSVM scoring not implemented in scaffold")

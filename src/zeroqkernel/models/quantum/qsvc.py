"""QSVC-style classifier wrapper for precomputed quantum kernels."""

from __future__ import annotations


class QSVCModel:
    """Skeleton wrapper for binary classification experiments."""

    def __init__(self, c: float = 1.0) -> None:
        self.c = c

    def fit(self, k_train, y_train):
        _ = k_train
        _ = y_train
        raise NotImplementedError("QSVC fitting not implemented in scaffold")

    def predict(self, k_eval):
        _ = k_eval
        raise NotImplementedError("QSVC predict not implemented in scaffold")

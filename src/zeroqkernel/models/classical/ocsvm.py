"""One-class SVM baseline wrapper."""

from __future__ import annotations

import numpy as np


class OCSVMBaseline:
    """Wrapper for one-class SVM with a similarity-score fallback."""

    def __init__(self, nu: float = 0.05) -> None:
        self.nu = nu
        self._model = None
        self._fallback_to_similarity = False

    def fit(self, k_train):
        try:
            from sklearn.svm import OneClassSVM
        except ImportError:
            self._fallback_to_similarity = True
            self._model = None
            return self

        self._model = OneClassSVM(kernel="precomputed", nu=self.nu)
        self._model.fit(k_train)
        self._fallback_to_similarity = False
        return self

    def score_samples(self, k_eval):
        if self._fallback_to_similarity:
            # Match the sklearn path: larger values mean more anomalous.
            return 1.0 - np.asarray(k_eval).mean(axis=1)

        if self._model is None:
            raise RuntimeError("Call fit() before score_samples().")

        decision = self._model.decision_function(k_eval)
        return -np.asarray(decision, dtype=np.float64).reshape(-1)

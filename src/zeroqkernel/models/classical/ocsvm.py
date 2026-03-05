"""원클래스 SVM 베이스라인 래퍼."""

from __future__ import annotations

import numpy as np


class OCSVMBaseline:
    """원클래스 SVM 래퍼. 필요 시 유사도 점수 기반 폴백을 사용한다."""

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
            # 사이킷런 경로와 맞춘다: 값이 클수록 더 이상으로 본다.
            return 1.0 - np.asarray(k_eval).mean(axis=1)

        if self._model is None:
            raise RuntimeError("Call fit() before score_samples().")

        decision = self._model.decision_function(k_eval)
        return -np.asarray(decision, dtype=np.float64).reshape(-1)

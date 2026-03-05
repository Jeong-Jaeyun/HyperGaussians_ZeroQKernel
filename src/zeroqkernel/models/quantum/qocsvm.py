"""사전 계산된 양자 커널을 사용하는 원클래스 이상 탐지 모델."""

from __future__ import annotations

import numpy as np


class QuantumOneClassModel:
    """사전 계산 커널용 원클래스 SVM 래퍼."""

    def __init__(self, nu: float = 0.05) -> None:
        self.nu = nu
        self._model = None
        self._fallback_to_similarity = False

    def fit(self, k_train):
        # k_train은 QuantumKernel.gram(x_train) 결과를 그대로 받는다고 가정한다.
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
        # k_eval은 QuantumKernel.gram(x_eval, x_train) 결과다.
        # 반환값은 이상 점수이며, 값이 클수록 더 이상으로 해석한다.
        if self._fallback_to_similarity:
            return 1.0 - np.asarray(k_eval).mean(axis=1)

        if self._model is None:
            raise RuntimeError("Call fit() before score_samples().")

        decision = self._model.decision_function(k_eval)
        return -np.asarray(decision, dtype=np.float64).reshape(-1)

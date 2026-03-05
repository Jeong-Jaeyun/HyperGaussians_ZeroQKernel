"""QSVC-style classifier wrapper for precomputed quantum kernels."""

from __future__ import annotations


class QSVCModel:
    """Skeleton wrapper for binary classification experiments."""

    def __init__(self, c: float = 1.0) -> None:
        self.c = c

    def fit(self, k_train, y_train):
        # 향후 binary 분기를 만들 때는 QuantumKernel.gram(x_train)으로
        # 만든 precomputed kernel과 레이블을 여기로 넘기면 된다.
        _ = k_train
        _ = y_train
        raise NotImplementedError("QSVC fitting not implemented in scaffold")

    def predict(self, k_eval):
        # 예측 시에는 QuantumKernel.gram(x_eval, x_train) 결과를 사용한다.
        _ = k_eval
        raise NotImplementedError("QSVC predict not implemented in scaffold")

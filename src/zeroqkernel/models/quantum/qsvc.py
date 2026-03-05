"""사전 계산 양자 커널용 QSVC 스타일 분류기 래퍼."""

from __future__ import annotations


class QSVCModel:
    """이진 분류 실험을 위한 골격 래퍼."""

    def __init__(self, c: float = 1.0) -> None:
        self.c = c

    def fit(self, k_train, y_train):
        # 향후 이진 분기를 만들 때는 QuantumKernel.gram(x_train)으로
        # 만든 사전 계산 커널과 레이블을 여기로 넘기면 된다.
        _ = k_train
        _ = y_train
        raise NotImplementedError("QSVC fitting not implemented in scaffold")

    def predict(self, k_eval):
        # 예측 시에는 QuantumKernel.gram(x_eval, x_train) 결과를 사용한다.
        _ = k_eval
        raise NotImplementedError("QSVC predict not implemented in scaffold")

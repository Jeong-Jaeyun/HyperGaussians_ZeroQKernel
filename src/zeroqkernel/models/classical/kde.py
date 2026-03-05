"""커널 밀도 추정 베이스라인 래퍼."""

from __future__ import annotations


class KDEBaseline:
    """밀도 기반 이상 탐지 베이스라인 골격."""

    def __init__(self, bandwidth: float = 1.0) -> None:
        self.bandwidth = bandwidth

    def fit(self, x_train):
        _ = x_train
        raise NotImplementedError("KDE baseline not implemented in scaffold")

    def score_samples(self, x_eval):
        _ = x_eval
        raise NotImplementedError("KDE baseline not implemented in scaffold")

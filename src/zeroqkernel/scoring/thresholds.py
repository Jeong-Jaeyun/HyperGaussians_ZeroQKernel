"""임계값 보정 유틸리티."""

from __future__ import annotations

import numpy as np


def calibrate_quantile(scores: np.ndarray, false_positive_rate: float = 0.01) -> float:
    """정상 점수의 상위 분위수로 임계값을 설정한다."""
    q = 1.0 - false_positive_rate
    return float(np.quantile(scores, q))


def apply_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    """1이 이상을 의미하는 이진 판정을 반환한다."""
    return (scores >= threshold).astype(int)

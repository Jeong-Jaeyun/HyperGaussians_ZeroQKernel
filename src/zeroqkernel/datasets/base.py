"""실험 전반에서 사용하는 데이터셋 인터페이스."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class DatasetBundle:
    """학습/검증/테스트 배열을 담는 컨테이너."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray

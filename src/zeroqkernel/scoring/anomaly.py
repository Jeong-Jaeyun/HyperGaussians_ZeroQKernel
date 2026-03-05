"""커널 기반 IDS용 이상 점수 생성기."""

from __future__ import annotations

import numpy as np


def mean_kernel_similarity(k_eval_train: np.ndarray) -> np.ndarray:
    """값이 클수록 학습 정상 샘플과 더 유사함을 의미한다."""
    return k_eval_train.mean(axis=1)


def mmd_score(k_eval_train: np.ndarray) -> np.ndarray:
    """플레이스홀더 프록시이며, 전체 MMD 정식으로 교체해야 한다."""
    sim = mean_kernel_similarity(k_eval_train)
    return 1.0 - sim

"""재현 가능한 실험을 위한 난수 시드 제어."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Python과 NumPy 시드를 설정하고, 필요하면 결정적 해시 시드도 강제한다."""
    random.seed(seed)
    np.random.seed(seed)
    if deterministic:
        os.environ["PYTHONHASHSEED"] = str(seed)

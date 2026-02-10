"""Random seed control for reproducible experiments."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set python and numpy seeds; optionally force deterministic hash seed."""
    random.seed(seed)
    np.random.seed(seed)
    if deterministic:
        os.environ["PYTHONHASHSEED"] = str(seed)

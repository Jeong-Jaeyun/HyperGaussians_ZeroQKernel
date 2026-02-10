"""Quantum kernel interfaces and gram matrix generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class QuantumKernel:
    """Kernel API: single-pair kernel and batch gram matrix."""

    def __init__(self, feature_map, cache_dir: str | Path | None = None) -> None:
        self.feature_map = feature_map
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def kernel(self, x, x_prime) -> float:
        _ = x
        _ = x_prime
        raise NotImplementedError("Single quantum kernel evaluation not implemented")

    def gram(self, x: np.ndarray, x2: np.ndarray | None = None) -> np.ndarray:
        _ = x
        _ = x2
        raise NotImplementedError("Gram matrix evaluation not implemented")

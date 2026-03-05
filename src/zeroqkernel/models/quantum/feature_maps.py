"""Toy hyper-gaussian quantum feature map scaffold.

This is intentionally lightweight. It exposes the important decisions you will
eventually replace with a real circuit construction:

1. How to compress high-dimensional tabular features into ``n_qubits`` slots.
2. How to turn those reduced features into hyper-gaussian amplitude weights.
3. How to define per-layer rotation and phase schedules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class HyperGaussianFeatureMapConfig:
    n_qubits: int
    depth: int
    sigma: float
    order: int
    phase_scale: float
    entanglement_pattern: str = "ring"
    data_reupload: bool = True


@dataclass(slots=True)
class EncodedHyperGaussianSample:
    # 이 구조체가 QuantumKernel로 넘어가서 커널 유사도 계산의 재료가 된다.
    reduced_x: np.ndarray
    envelope: np.ndarray
    rotation_layers: np.ndarray
    phase_layers: np.ndarray
    entangling_pairs: list[tuple[int, int]]


class HyperGaussianFeatureMap:
    """Toy feature-map builder for hyper-gaussian-inspired encodings.

    The output is not a Qiskit circuit. It is a compact encoding spec that is
    easy to inspect and replace with a real backend-specific circuit later.
    """

    def __init__(self, config: HyperGaussianFeatureMapConfig) -> None:
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        if self.config.n_qubits <= 0:
            raise ValueError("n_qubits must be positive")
        if self.config.depth <= 0:
            raise ValueError("depth must be positive")
        if self.config.sigma <= 0:
            raise ValueError("sigma must be positive")
        if self.config.order <= 0:
            raise ValueError("order must be positive")
        if self.config.entanglement_pattern not in {"ring", "line", "none"}:
            raise ValueError(
                "entanglement_pattern must be one of: ring, line, none"
            )

    def _to_1d_vector(self, x) -> np.ndarray:
        vector = np.asarray(x, dtype=np.float64).reshape(-1)
        if vector.size == 0:
            raise ValueError("Input sample must contain at least one feature")
        return vector

    def reduce_features(self, x) -> np.ndarray:
        """Map an arbitrary-length feature vector down to ``n_qubits`` slots.

        This uses deterministic chunk-mean pooling. It is simple on purpose,
        and it is one of the first places you would replace for a better design.
        """
        vector = self._to_1d_vector(x)
        n_qubits = self.config.n_qubits

        if vector.size == n_qubits:
            return vector

        if vector.size < n_qubits:
            # 특성 수가 큐비트 수보다 적으면 뒤를 0으로 채워 고정 길이를 맞춘다.
            reduced = np.zeros(n_qubits, dtype=np.float64)
            reduced[: vector.size] = vector
            return reduced

        edges = np.linspace(0, vector.size, num=n_qubits + 1, dtype=int)
        reduced = np.empty(n_qubits, dtype=np.float64)
        for idx in range(n_qubits):
            start = int(edges[idx])
            end = int(edges[idx + 1])
            if end <= start:
                end = min(start + 1, vector.size)
            reduced[idx] = float(vector[start:end].mean())
        return reduced

    def hyper_gaussian_envelope(self, reduced_x: np.ndarray) -> np.ndarray:
        """Compute a hyper-gaussian-style envelope over reduced features."""
        sigma = max(float(self.config.sigma), 1e-8)
        order = int(self.config.order)
        scaled = np.abs(reduced_x) / sigma
        return np.exp(-np.power(scaled, 2 * order))

    def build_rotation_layers(
        self, reduced_x: np.ndarray, envelope: np.ndarray
    ) -> np.ndarray:
        """Build per-layer toy rotation angles.

        Replace this rule with your actual gate-parameter schedule when you move
        from toy code to a real feature map.
        """
        base = np.pi * np.tanh(reduced_x * envelope)
        layers = np.empty((self.config.depth, self.config.n_qubits), dtype=np.float64)
        for layer_idx in range(self.config.depth):
            scale = 1.0 if not self.config.data_reupload else 1.0 + 0.1 * layer_idx
            layers[layer_idx] = scale * base
        return layers

    def build_phase_layers(
        self, reduced_x: np.ndarray, envelope: np.ndarray
    ) -> np.ndarray:
        """Build per-layer toy phase angles.

        This is another deliberate simplification. Keep the shape, swap the rule.
        """
        base = float(self.config.phase_scale) * reduced_x * envelope
        layers = np.empty((self.config.depth, self.config.n_qubits), dtype=np.float64)
        for layer_idx in range(self.config.depth):
            phase_gain = float(layer_idx + 1) / float(self.config.depth)
            if not self.config.data_reupload and layer_idx > 0:
                phase_gain = 0.0
            layers[layer_idx] = phase_gain * base
        return layers

    def build_entangling_pairs(self) -> list[tuple[int, int]]:
        """Return the pair list for the chosen toy entanglement topology."""
        n_qubits = self.config.n_qubits
        pattern = self.config.entanglement_pattern

        if pattern == "none" or n_qubits < 2:
            return []
        if pattern == "line":
            return [(idx, idx + 1) for idx in range(n_qubits - 1)]

        pairs = [(idx, idx + 1) for idx in range(n_qubits - 1)]
        pairs.append((n_qubits - 1, 0))
        return pairs

    def encode(self, x) -> EncodedHyperGaussianSample:
        """Produce a backend-agnostic encoding spec for one sample."""
        # 연결 흐름:
        # 원본 샘플 1개 -> 차원 축소 -> 하이퍼 가우시안 envelope ->
        # rotation/phase 레이어 생성 -> entanglement 구조 결정 ->
        # EncodedHyperGaussianSample 반환 -> QuantumKernel.kernel()에서 사용
        reduced_x = self.reduce_features(x)
        envelope = self.hyper_gaussian_envelope(reduced_x)
        rotation_layers = self.build_rotation_layers(reduced_x, envelope)
        phase_layers = self.build_phase_layers(reduced_x, envelope)
        entangling_pairs = self.build_entangling_pairs()

        return EncodedHyperGaussianSample(
            reduced_x=reduced_x,
            envelope=envelope,
            rotation_layers=rotation_layers,
            phase_layers=phase_layers,
            entangling_pairs=entangling_pairs,
        )

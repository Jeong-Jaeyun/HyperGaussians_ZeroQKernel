"""하이퍼 가우시안 기반 장난감 양자 피처 맵 골격 코드.

의도적으로 가볍게 유지한 구현이며, 실제 회로 구성으로 대체할 핵심 결정 지점을 드러낸다.

1. 고차원 테이블 특성을 ``n_qubits`` 슬롯으로 압축하는 방법.
2. 축소된 특성을 하이퍼 가우시안 진폭 가중치로 바꾸는 방법.
3. 레이어별 회전/위상 스케줄을 정의하는 방법.
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
    """하이퍼 가우시안 영감 인코딩을 위한 장난감 피처 맵 생성기.

    출력은 Qiskit 회로가 아니라, 나중에 백엔드별 실제 회로로 교체하기 쉬운
    간결한 인코딩 명세다.
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
        """임의 길이 특성 벡터를 ``n_qubits`` 슬롯으로 축소한다.

        결정적 청크 평균 풀링을 사용한다. 의도적으로 단순하게 만들었고,
        더 나은 설계에서는 가장 먼저 교체할 수 있는 지점이다.
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
        """축소된 특성 위에서 하이퍼 가우시안 형태의 엔벌로프를 계산한다."""
        sigma = max(float(self.config.sigma), 1e-8)
        order = int(self.config.order)
        scaled = np.abs(reduced_x) / sigma
        return np.exp(-np.power(scaled, 2 * order))

    def build_rotation_layers(
        self, reduced_x: np.ndarray, envelope: np.ndarray
    ) -> np.ndarray:
        """레이어별 장난감 회전 각도를 만든다.

        장난감 코드에서 실제 피처 맵으로 전환할 때는 이 규칙을
        실제 게이트 파라미터 스케줄로 교체하면 된다.
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
        """레이어별 장난감 위상 각도를 만든다.

        이것도 의도적인 단순화다. 텐서 형태는 유지하고 규칙만 바꾸면 된다.
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
        """선택한 장난감 얽힘 토폴로지의 페어 목록을 반환한다."""
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
        """샘플 하나에 대한 백엔드 독립 인코딩 명세를 생성한다."""
        # 연결 흐름:
        # 원본 샘플 1개 -> 차원 축소 -> 하이퍼 가우시안 엔벌로프 ->
        # 회전/위상 레이어 생성 -> 얽힘 구조 결정 ->
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

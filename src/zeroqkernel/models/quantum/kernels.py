"""Toy quantum-kernel scaffold for hyper-gaussian encodings.

The default implementation uses an analytic overlap proxy on top of the feature
map spec. That keeps the code small and inspectable, while still making the
important replacement point explicit: swap the overlap proxy for a real circuit
execution or statevector overlap.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .feature_maps import EncodedHyperGaussianSample, HyperGaussianFeatureMap


class QuantumKernel:
    """Backend-agnostic quantum-kernel interface with a toy overlap proxy."""

    def __init__(
        self,
        feature_map: HyperGaussianFeatureMap,
        cache_dir: str | Path | None = None,
        use_analytic_proxy: bool = True,
    ) -> None:
        self.feature_map = feature_map
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_analytic_proxy = use_analytic_proxy
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        # 이 객체는 HyperGaussianFeatureMap.encode() 결과를 받아
        # pairwise kernel 값과 Gram matrix를 만드는 중간 계층이다.

    def _to_state_proxy(self, encoded: EncodedHyperGaussianSample) -> np.ndarray:
        """Convert an encoded sample into a small complex proxy state.

        This is not a real quantum state preparation. It is a deterministic
        placeholder that preserves the main ingredients you still care about:
        envelope, layered rotations, layered phases, and entanglement topology.
        """
        rotation_signal = encoded.rotation_layers.mean(axis=0)
        phase_signal = encoded.phase_layers.sum(axis=0)

        state = encoded.envelope.astype(np.complex128, copy=True)
        state *= np.exp(1j * phase_signal)
        state *= np.cos(0.5 * rotation_signal) + 1j * np.sin(0.5 * rotation_signal)

        for left, right in encoded.entangling_pairs:
            coupling = 0.25 * (rotation_signal[left] + rotation_signal[right])
            state[left] *= np.exp(1j * coupling)
            state[right] *= np.exp(-1j * coupling)

        norm = float(np.linalg.norm(state))
        if norm <= 1e-12:
            return np.full(state.shape, 1.0 / np.sqrt(state.size), dtype=np.complex128)
        return state / norm

    def estimate_overlap_proxy(
        self,
        encoded_x: EncodedHyperGaussianSample,
        encoded_x_prime: EncodedHyperGaussianSample,
    ) -> float:
        """Toy overlap estimate.

        This is the main blank to replace when you move to an actual quantum
        kernel: use a circuit backend, sampler, or statevector overlap here.
        """
        psi = self._to_state_proxy(encoded_x)
        phi = self._to_state_proxy(encoded_x_prime)
        overlap = np.abs(np.vdot(psi, phi)) ** 2
        return float(np.clip(overlap.real, 0.0, 1.0))

    def kernel(self, x, x_prime) -> float:
        # 샘플 2개를 feature_map으로 각각 encode한 뒤, 그 결과의 overlap으로
        # 단일 커널 값을 만든다.
        if not self.use_analytic_proxy:
            raise NotImplementedError(
                "Replace estimate_overlap_proxy() with a backend-specific overlap "
                "and then enable that path here."
            )

        encoded_x = self.feature_map.encode(x)
        encoded_x_prime = self.feature_map.encode(x_prime)
        return self.estimate_overlap_proxy(encoded_x, encoded_x_prime)

    def gram(self, x: np.ndarray, x2: np.ndarray | None = None) -> np.ndarray:
        # 여기서 만든 Gram matrix가 QuantumOneClassModel.fit()/score_samples()의
        # 입력이 된다. 즉 feature map과 학습 모델 사이를 연결하는 핵심 포인트다.
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array")

        if x2 is None:
            n_rows = x.shape[0]
            gram = np.empty((n_rows, n_rows), dtype=np.float64)
            for row in range(n_rows):
                gram[row, row] = self.kernel(x[row], x[row])
                for col in range(row + 1, n_rows):
                    value = self.kernel(x[row], x[col])
                    gram[row, col] = value
                    gram[col, row] = value
            return gram

        x2 = np.asarray(x2, dtype=np.float64)
        if x2.ndim != 2:
            raise ValueError("x2 must be a 2D array")

        gram = np.empty((x.shape[0], x2.shape[0]), dtype=np.float64)
        for row in range(x.shape[0]):
            for col in range(x2.shape[0]):
                gram[row, col] = self.kernel(x[row], x2[col])
        return gram

"""하이퍼 가우시안 인코딩용 양자 커널 골격 코드.

기본 구현은 피처 맵 명세 위에 해석적 오버랩 프록시를 사용한다.
이렇게 하면 코드를 작고 읽기 쉽게 유지하면서도 핵심 교체 지점을 명확히 드러낼 수 있다.
즉, 프록시 오버랩 계산을 실제 회로 실행 또는 상태벡터 오버랩으로 바꾸면 된다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .feature_maps import EncodedHyperGaussianSample, HyperGaussianFeatureMap


class QuantumKernel:
    """오버랩 프록시를 갖는 백엔드 독립 양자 커널 인터페이스."""

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
        # 쌍별 커널 값과 그램 행렬을 만드는 중간 계층이다.

    def _to_state_proxy(self, encoded: EncodedHyperGaussianSample) -> np.ndarray:
        """인코딩된 샘플을 작은 복소수 프록시 상태로 변환한다.

        실제 양자 상태 준비는 아니며, 다음 핵심 요소를 보존하는
        결정적 플레이스홀더 구현이다:
        엔벌로프, 레이어 회전, 레이어 위상, 얽힘 토폴로지.
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
        """오버랩 추정값을 계산한다.

        실제 양자 커널로 전환할 때 가장 먼저 교체할 핵심 지점이다.
        이 부분에 회로 백엔드, 샘플러, 상태벡터 오버랩 계산을 넣으면 된다.
        """
        psi = self._to_state_proxy(encoded_x)
        phi = self._to_state_proxy(encoded_x_prime)
        overlap = np.abs(np.vdot(psi, phi)) ** 2
        return float(np.clip(overlap.real, 0.0, 1.0))

    def kernel(self, x, x_prime) -> float:
        # 샘플 2개를 feature_map으로 각각 encode한 뒤, 그 결과의 오버랩으로
        # 단일 커널 값을 만든다.
        if not self.use_analytic_proxy:
            raise NotImplementedError(
                "estimate_overlap_proxy()를 백엔드별 오버랩 계산으로 바꾼 뒤 "
                "여기 경로를 활성화하세요."
            )

        encoded_x = self.feature_map.encode(x)
        encoded_x_prime = self.feature_map.encode(x_prime)
        return self.estimate_overlap_proxy(encoded_x, encoded_x_prime)

    def gram(self, x: np.ndarray, x2: np.ndarray | None = None) -> np.ndarray:
        # 여기서 만든 그램 행렬이 QuantumOneClassModel.fit()/score_samples()의
        # 입력이 된다. 즉 피처 맵과 학습 모델 사이를 연결하는 핵심 포인트다.
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

"""효율 추적 유틸리티(지연시간, 샷 수, 커널 비용)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EfficiencyMetrics:
    kernel_build_seconds: float | None = None
    inference_seconds: float | None = None
    shots: int | None = None

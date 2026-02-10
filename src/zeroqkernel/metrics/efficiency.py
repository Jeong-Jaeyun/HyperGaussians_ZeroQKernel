"""Efficiency tracking utilities (latency, shots, kernel cost)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EfficiencyMetrics:
    kernel_build_seconds: float | None = None
    inference_seconds: float | None = None
    shots: int | None = None

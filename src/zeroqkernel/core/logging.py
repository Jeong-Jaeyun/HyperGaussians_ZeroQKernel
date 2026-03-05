"""실험별 디렉터리에 설정과 지표를 기록하는 실험 로거."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import ensure_dir, save_json


class ExperimentLogger:
    """재현 가능한 실행 산출물을 위한 파일 시스템 기반 로거."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = ensure_dir(run_dir)

    def log_config(self, config: dict[str, Any]) -> None:
        save_json(self.run_dir / "config.json", config)

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        save_json(self.run_dir / "metrics.json", metrics)

    def log_note(self, text: str) -> None:
        (self.run_dir / "notes.txt").write_text(text, encoding="utf-8")

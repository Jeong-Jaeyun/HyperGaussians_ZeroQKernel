"""Experiment logger writing config and metrics to per-run directories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import ensure_dir, save_json


class ExperimentLogger:
    """File-system based logger for reproducible run artifacts."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = ensure_dir(run_dir)

    def log_config(self, config: dict[str, Any]) -> None:
        save_json(self.run_dir / "config.json", config)

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        save_json(self.run_dir / "metrics.json", metrics)

    def log_note(self, text: str) -> None:
        (self.run_dir / "notes.txt").write_text(text, encoding="utf-8")

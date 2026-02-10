"""Dataset loading entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_dataset(config: dict[str, Any]) -> tuple:
    """Load raw dataset and labels according to dataset config.

    TODO:
    - support CSV, parquet, and prebuilt tensor formats
    - standardize return type for downstream preprocess/split
    """
    source = config.get("source", {})
    path = Path(source.get("path", ""))
    if not path.exists():
        raise FileNotFoundError(f"Dataset source path does not exist: {path}")
    raise NotImplementedError("Dataset loader skeleton only")

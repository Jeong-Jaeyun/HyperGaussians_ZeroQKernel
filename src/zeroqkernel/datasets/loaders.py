"""Dataset loading entry points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(slots=True)
class LoadedDataset:
    """Raw tabular dataset loaded from one or more source files."""

    features: pd.DataFrame
    labels: pd.Series
    feature_names: list[str]
    source_files: list[str]


def _resolve_csv_files(path: Path, pattern: str, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() == ".csv" else []
    if not path.is_dir():
        return []
    files = path.rglob(pattern) if recursive else path.glob(pattern)
    return sorted(p for p in files if p.is_file())


def _normalize_columns(columns) -> list[str]:
    return [str(col).strip() for col in columns]


def _find_label_column(columns: list[str], configured_name: str) -> str:
    if configured_name in columns:
        return configured_name

    normalized = {col.strip().casefold(): col for col in columns}
    target = configured_name.strip().casefold()
    if target in normalized:
        return normalized[target]
    if "label" in normalized:
        return normalized["label"]
    raise KeyError(
        f"Label column not found. Configured={configured_name!r}, available={columns}"
    )


def load_dataset(config: dict[str, Any]) -> LoadedDataset:
    """Load raw dataset and labels according to dataset config.

    Supports single-file CSV or directory-of-CSV layouts.
    """
    source = config.get("source", {})
    path = Path(source.get("path", ""))
    if not path.exists():
        raise FileNotFoundError(f"Dataset source path does not exist: {path}")

    file_format = str(source.get("format", "csv")).lower()
    if file_format != "csv":
        raise ValueError(f"Unsupported source format: {file_format}")

    pattern = str(source.get("pattern", "*.csv"))
    recursive = bool(source.get("recursive", False))
    label_column = str(source.get("label_column", "Label"))
    max_rows = source.get("max_rows")
    if max_rows is not None:
        max_rows = int(max_rows)
        if max_rows <= 0:
            raise ValueError("source.max_rows must be positive when provided")

    csv_files = _resolve_csv_files(path, pattern=pattern, recursive=recursive)
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found at {path} with pattern={pattern!r} recursive={recursive}"
        )

    frames: list[pd.DataFrame] = []
    remaining = max_rows
    used_files: list[str] = []
    for csv_file in csv_files:
        if remaining is not None and remaining <= 0:
            break

        frame = pd.read_csv(csv_file, nrows=remaining, low_memory=False)
        if frame.empty:
            continue

        frame.columns = _normalize_columns(frame.columns)
        detected_label_column = _find_label_column(list(frame.columns), label_column)
        if detected_label_column != "Label":
            frame = frame.rename(columns={detected_label_column: "Label"})
        frames.append(frame)
        used_files.append(str(csv_file))

        if remaining is not None:
            remaining -= len(frame)

    if not frames:
        raise ValueError("CSV sources were found, but all loaded frames are empty")

    merged = pd.concat(frames, axis=0, ignore_index=True)
    merged.columns = _normalize_columns(merged.columns)
    if "Label" not in merged.columns:
        detected = _find_label_column(list(merged.columns), label_column)
        merged = merged.rename(columns={detected: "Label"})

    labels = merged.pop("Label")
    feature_names = _normalize_columns(merged.columns)
    merged.columns = feature_names

    return LoadedDataset(
        features=merged,
        labels=labels,
        feature_names=feature_names,
        source_files=used_files,
    )

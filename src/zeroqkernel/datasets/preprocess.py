"""Preprocessing pipeline for IDS features."""

from __future__ import annotations

from typing import Any


def preprocess_features(x, config: dict[str, Any]):
    """Apply normalization and optional window/encoding transformations."""
    _ = config
    return x


def preprocess_dataset(raw_data, config: dict[str, Any]):
    """Convert raw dataset output into model-ready tensors.

    TODO:
    - numeric/categorical handling
    - missing value strategy
    - feature ordering lock for reproducibility
    """
    _ = raw_data
    _ = config
    raise NotImplementedError("Preprocess skeleton only")

"""IDS 특성을 위한 전처리 파이프라인."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from zeroqkernel.datasets.loaders import LoadedDataset


@dataclass(slots=True)
class PreprocessedDataset:
    x: np.ndarray
    y: np.ndarray
    raw_labels: np.ndarray
    feature_names: list[str]
    stats: dict[str, Any]


def _parse_clip_quantiles(pre_cfg: dict[str, Any]) -> tuple[bool, float, float]:
    qcfg = pre_cfg.get("clip_quantiles")
    if qcfg is None:
        return False, 0.0, 1.0

    if isinstance(qcfg, dict):
        enabled = bool(qcfg.get("enabled", True))
        low = float(qcfg.get("low", 0.001))
        high = float(qcfg.get("high", 0.999))
    elif isinstance(qcfg, (list, tuple)) and len(qcfg) == 2:
        enabled = True
        low = float(qcfg[0])
        high = float(qcfg[1])
    else:
        raise ValueError("preprocess.clip_quantiles must be dict or [low, high]")

    if not 0.0 <= low < high <= 1.0:
        raise ValueError("clip quantiles must satisfy 0 <= low < high <= 1")
    return enabled, low, high


def _make_binary_labels(labels: np.ndarray, normal_label: str) -> np.ndarray:
    cleaned = np.char.strip(labels.astype(str))
    cleaned_folded = np.char.lower(cleaned)
    normal_folded = normal_label.strip().casefold()
    return (cleaned_folded != normal_folded).astype(np.int8)


def preprocess_features(
    x: np.ndarray, config: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """정규화와 후처리 클리핑 변환을 적용한다."""
    normalize = str(config.get("normalize", "zscore")).lower()
    eps = float(config.get("eps", 1e-8))
    stats: dict[str, Any] = {"normalize": normalize}

    if normalize in {"none", "identity"}:
        pass
    elif normalize == "zscore":
        mean = x.mean(axis=0, keepdims=True, dtype=np.float64)
        std = x.std(axis=0, keepdims=True, dtype=np.float64)
        zero_std_mask = std < eps
        std[zero_std_mask] = 1.0
        x = (x - mean) / std
        stats["zero_std_features"] = int(zero_std_mask.sum())
    elif normalize == "minmax":
        low = x.min(axis=0, keepdims=True)
        high = x.max(axis=0, keepdims=True)
        span = high - low
        zero_span_mask = span < eps
        span[zero_span_mask] = 1.0
        x = (x - low) / span
        stats["zero_span_features"] = int(zero_span_mask.sum())
    else:
        raise ValueError(f"Unsupported preprocess.normalize mode: {normalize}")

    post_clip_abs = config.get("post_clip_abs")
    if post_clip_abs is not None:
        post_clip_abs = float(post_clip_abs)
        if post_clip_abs <= 0:
            raise ValueError("preprocess.post_clip_abs must be > 0")
        x = np.clip(x, -post_clip_abs, post_clip_abs)
        stats["post_clip_abs"] = post_clip_abs
    return x, stats


def preprocess_dataset(
    raw_data: LoadedDataset, config: dict[str, Any]
) -> PreprocessedDataset:
    """원본 데이터셋 출력을 모델 입력용 배열로 변환한다."""
    pre_cfg = config.get("preprocess", {})
    labels_cfg = config.get("labels", {})
    missing_cfg = pre_cfg.get("missing", {})

    x_df = raw_data.features.copy()
    x_df.columns = [str(col).strip() for col in x_df.columns]
    x_df = x_df.replace([np.inf, -np.inf], np.nan)
    x_numeric = x_df.apply(pd.to_numeric, errors="coerce")

    missing_before = int(x_numeric.isna().sum().sum())
    drop_threshold = float(missing_cfg.get("drop_if_missing_ratio_above", 0.98))
    if not 0.0 <= drop_threshold <= 1.0:
        raise ValueError("missing.drop_if_missing_ratio_above must be in [0, 1]")

    missing_ratio = x_numeric.isna().mean(axis=0)
    dropped_cols = missing_ratio[missing_ratio > drop_threshold].index.tolist()
    if dropped_cols:
        x_numeric = x_numeric.drop(columns=dropped_cols)

    impute_strategy = str(missing_cfg.get("impute", "median")).lower()
    if impute_strategy == "median":
        fill_values = x_numeric.median(axis=0, numeric_only=True)
    elif impute_strategy == "mean":
        fill_values = x_numeric.mean(axis=0, numeric_only=True)
    elif impute_strategy == "zero":
        fill_values = pd.Series(0.0, index=x_numeric.columns)
    else:
        raise ValueError(f"Unsupported missing.impute strategy: {impute_strategy}")

    fill_values = fill_values.reindex(x_numeric.columns).fillna(0.0)
    x_numeric = x_numeric.fillna(fill_values)

    clip_enabled, q_low, q_high = _parse_clip_quantiles(pre_cfg)
    if clip_enabled:
        lower = x_numeric.quantile(q_low, axis=0)
        upper = x_numeric.quantile(q_high, axis=0)
        x_numeric = x_numeric.clip(lower=lower, upper=upper, axis=1)

    x = x_numeric.to_numpy(dtype=np.float64, copy=True)
    x, norm_stats = preprocess_features(x, pre_cfg)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(x).all():
        raise ValueError("Preprocessing produced non-finite features")

    dtype_name = str(pre_cfg.get("dtype", "float32")).lower()
    if dtype_name == "float32":
        x = x.astype(np.float32, copy=False)
    elif dtype_name == "float64":
        x = x.astype(np.float64, copy=False)
    else:
        raise ValueError("preprocess.dtype must be float32 or float64")

    raw_labels = raw_data.labels.astype(str).str.strip().to_numpy(dtype="U")
    normal_label = str(labels_cfg.get("normal_label", "BENIGN"))
    y = _make_binary_labels(raw_labels, normal_label=normal_label)

    stats: dict[str, Any] = {
        "rows": int(x.shape[0]),
        "features_in": int(len(raw_data.feature_names)),
        "features_out": int(x.shape[1]),
        "dropped_features": dropped_cols,
        "missing_values_before_impute": missing_before,
        "missing_values_after_impute": int(np.isnan(x).sum()),
        "normal_label": normal_label,
        "normal_count": int((y == 0).sum()),
        "anomaly_count": int((y == 1).sum()),
        "source_files": list(raw_data.source_files),
        "impute_strategy": impute_strategy,
    }
    if clip_enabled:
        stats["clip_quantiles"] = {"low": q_low, "high": q_high}
    stats.update(norm_stats)

    feature_names = [name for name in raw_data.feature_names if name not in dropped_cols]
    return PreprocessedDataset(
        x=x,
        y=y,
        raw_labels=raw_labels,
        feature_names=feature_names,
        stats=stats,
    )

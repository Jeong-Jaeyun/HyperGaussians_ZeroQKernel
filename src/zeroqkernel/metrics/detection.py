"""탐지 지표 인터페이스."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class DetectionMetrics:
    auroc: float | None = None
    auprc: float | None = None
    tpr_at_fpr_1: float | None = None
    eer: float | None = None


def compute_detection_metrics(y_true, scores) -> DetectionMetrics:
    """AUROC, AUPRC, TPR@FPR=1%, EER를 계산한다."""
    y_true = np.asarray(y_true, dtype=np.int8).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)

    if y_true.shape[0] != scores.shape[0]:
        raise ValueError("y_true and scores must have the same number of rows")

    if y_true.size == 0 or np.unique(y_true).size < 2:
        return DetectionMetrics()

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
    except ImportError:
        return DetectionMetrics()

    auroc = float(roc_auc_score(y_true, scores))
    auprc = float(average_precision_score(y_true, scores))

    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1.0 - tpr

    at_target = fpr <= 0.01
    if np.any(at_target):
        tpr_at_fpr_1 = float(np.max(tpr[at_target]))
    else:
        closest_idx = int(np.argmin(np.abs(fpr - 0.01)))
        tpr_at_fpr_1 = float(tpr[closest_idx])

    eer_idx = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) * 0.5)

    return DetectionMetrics(
        auroc=auroc,
        auprc=auprc,
        tpr_at_fpr_1=tpr_at_fpr_1,
        eer=eer,
    )

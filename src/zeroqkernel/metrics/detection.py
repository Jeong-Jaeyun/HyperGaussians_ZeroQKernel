"""Detection metric interfaces."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DetectionMetrics:
    auroc: float | None = None
    auprc: float | None = None
    tpr_at_fpr_1: float | None = None
    eer: float | None = None


def compute_detection_metrics(y_true, scores) -> DetectionMetrics:
    """Compute AUROC, AUPRC, TPR@FPR=1%, and EER.

    TODO:
    - implement with sklearn or custom numerical routines
    """
    _ = y_true
    _ = scores
    return DetectionMetrics()

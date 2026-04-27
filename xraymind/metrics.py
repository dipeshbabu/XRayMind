"""Reliability and clinical-style evaluation metrics for XRayMind."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class BinaryMetrics:
    """Metrics for one binary pathology label."""

    label: str
    n: int
    prevalence: float
    threshold: float
    auroc: Optional[float]
    auprc: Optional[float]
    brier: float
    ece: float
    sensitivity: float
    specificity: float
    precision: float
    f1: float

    def as_dict(self) -> Dict[str, float | int | str | None]:
        return self.__dict__.copy()


def expected_calibration_error(
    y_true: Iterable[int], y_score: Iterable[float], n_bins: int = 10
) -> float:
    """Compute binary expected calibration error."""

    y_true_arr = np.asarray(list(y_true), dtype=float)
    y_score_arr = np.asarray(list(y_score), dtype=float)
    if len(y_true_arr) == 0:
        return float("nan")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (y_score_arr >= left) & (y_score_arr < right)
        if right == 1.0:
            mask = (y_score_arr >= left) & (y_score_arr <= right)
        if not np.any(mask):
            continue
        confidence = float(np.mean(y_score_arr[mask]))
        accuracy = float(np.mean(y_true_arr[mask]))
        ece += float(np.mean(mask)) * abs(accuracy - confidence)
    return ece


def specificity_score(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_pred_arr = np.asarray(list(y_pred), dtype=int)
    tn = int(np.sum((y_true_arr == 0) & (y_pred_arr == 0)))
    fp = int(np.sum((y_true_arr == 0) & (y_pred_arr == 1)))
    denom = tn + fp
    return float(tn / denom) if denom else float("nan")


def tune_threshold(
    y_true: Iterable[int],
    y_score: Iterable[float],
    objective: str = "f1",
    grid_size: int = 101,
) -> float:
    """Tune a binary decision threshold on validation data."""

    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_score_arr = np.asarray(list(y_score), dtype=float)
    thresholds = np.linspace(0.0, 1.0, grid_size)
    best_threshold = 0.5
    best_value = -np.inf

    for threshold in thresholds:
        y_pred = (y_score_arr >= threshold).astype(int)
        if objective == "f1":
            value = f1_score(y_true_arr, y_pred, zero_division=0)
        elif objective == "youden":
            sens = recall_score(y_true_arr, y_pred, zero_division=0)
            spec = specificity_score(y_true_arr, y_pred)
            value = sens + (0.0 if np.isnan(spec) else spec) - 1.0
        else:
            raise ValueError("objective must be 'f1' or 'youden'")
        if value > best_value:
            best_value = float(value)
            best_threshold = float(threshold)
    return best_threshold


def compute_binary_metrics(
    label: str,
    y_true: Iterable[int],
    y_score: Iterable[float],
    threshold: float = 0.5,
    n_bins: int = 10,
) -> BinaryMetrics:
    """Compute metrics for a single binary label."""

    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_score_arr = np.asarray(list(y_score), dtype=float)
    y_pred = (y_score_arr >= threshold).astype(int)
    has_both_classes = len(set(y_true_arr.tolist())) == 2

    return BinaryMetrics(
        label=label,
        n=int(len(y_true_arr)),
        prevalence=float(np.mean(y_true_arr)) if len(y_true_arr) else float("nan"),
        threshold=float(threshold),
        auroc=float(roc_auc_score(y_true_arr, y_score_arr)) if has_both_classes else None,
        auprc=float(average_precision_score(y_true_arr, y_score_arr)) if has_both_classes else None,
        brier=float(brier_score_loss(y_true_arr, y_score_arr)),
        ece=float(expected_calibration_error(y_true_arr, y_score_arr, n_bins=n_bins)),
        sensitivity=float(recall_score(y_true_arr, y_pred, zero_division=0)),
        specificity=float(specificity_score(y_true_arr, y_pred)),
        precision=float(precision_score(y_true_arr, y_pred, zero_division=0)),
        f1=float(f1_score(y_true_arr, y_pred, zero_division=0)),
    )

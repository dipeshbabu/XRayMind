"""Conformal prediction-set utilities for XRayMind.

These helpers provide split-conformal prediction sets for multi-label chest X-ray
probabilities. They are intended for research reliability analysis and human
review routing, not clinical deployment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import math
import numpy as np
import pandas as pd

POSITIVE_LABEL = "positive"
NEGATIVE_LABEL = "negative"


def binary_nonconformity_scores(y_true: Sequence[int], y_score: Sequence[float]) -> np.ndarray:
    """Return binary nonconformity scores for the observed class.

    For a positive case, the score is ``1 - p``. For a negative case, the score
    is ``p``. Smaller scores mean the model assigned higher probability to the
    observed class.
    """

    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(y_score, dtype=float), 0.0, 1.0)
    if len(y) != len(p):
        raise ValueError("y_true and y_score must have the same length")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("binary conformal prediction expects labels encoded as 0/1")
    return np.where(y == 1, 1.0 - p, p)


def conformal_quantile(scores: Sequence[float], alpha: float = 0.1) -> float:
    """Compute the finite-sample split-conformal threshold.

    Uses the standard ceil((n + 1) * (1 - alpha)) / n order statistic with a
    conservative cap at the largest calibration score.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    arr = np.asarray(scores, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        raise ValueError("scores must contain at least one finite value")
    sorted_scores = np.sort(arr)
    rank = int(math.ceil((len(sorted_scores) + 1) * (1.0 - alpha)))
    index = min(max(rank - 1, 0), len(sorted_scores) - 1)
    return float(sorted_scores[index])


def calibrate_conformal_thresholds(
    labels_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    image_column: str = "image",
    labels: Sequence[str] | None = None,
    alpha: float = 0.1,
) -> pd.DataFrame:
    """Calibrate one binary conformal threshold per pathology label."""

    merged = labels_df.merge(pred_df, on=image_column, suffixes=("_true", "_pred"))
    label_names = list(labels or [c for c in labels_df.columns if c != image_column and c in pred_df.columns])
    rows: List[Dict[str, Any]] = []
    for label in label_names:
        true_col = f"{label}_true" if f"{label}_true" in merged.columns else label
        pred_col = f"{label}_pred" if f"{label}_pred" in merged.columns else label
        if true_col not in merged.columns or pred_col not in merged.columns:
            continue
        valid = merged[[true_col, pred_col]].dropna()
        if valid.empty:
            continue
        y_true = valid[true_col].astype(int).to_numpy()
        y_score = valid[pred_col].astype(float).to_numpy()
        scores = binary_nonconformity_scores(y_true, y_score)
        rows.append(
            {
                "label": label,
                "alpha": float(alpha),
                "target_coverage": float(1.0 - alpha),
                "qhat": conformal_quantile(scores, alpha=alpha),
                "n_calibration": int(len(scores)),
                "mean_nonconformity": float(np.mean(scores)),
                "max_nonconformity": float(np.max(scores)),
            }
        )
    return pd.DataFrame(rows)


def _set_from_probability(probability: float, qhat: float) -> tuple[str, int, bool, bool]:
    """Return conformal prediction-set text, size, includes-negative, includes-positive."""

    p = float(np.clip(probability, 0.0, 1.0))
    include_negative = p <= qhat
    include_positive = (1.0 - p) <= qhat
    labels: List[str] = []
    if include_negative:
        labels.append(NEGATIVE_LABEL)
    if include_positive:
        labels.append(POSITIVE_LABEL)
    if not labels:
        set_text = "empty"
    else:
        set_text = "|".join(labels)
    return set_text, len(labels), bool(include_negative), bool(include_positive)


def apply_conformal_sets(
    pred_df: pd.DataFrame,
    thresholds_df: pd.DataFrame,
    image_column: str = "image",
) -> pd.DataFrame:
    """Add conformal prediction-set columns to a prediction frame."""

    out = pred_df.copy()
    for _, row in thresholds_df.iterrows():
        label = str(row["label"])
        if label not in out.columns:
            continue
        qhat = float(row["qhat"])
        set_values = [_set_from_probability(p, qhat) for p in out[label].astype(float).to_numpy()]
        out[f"{label}_conformal_set"] = [v[0] for v in set_values]
        out[f"{label}_conformal_set_size"] = [v[1] for v in set_values]
        out[f"{label}_conformal_includes_negative"] = [v[2] for v in set_values]
        out[f"{label}_conformal_includes_positive"] = [v[3] for v in set_values]
        out[f"{label}_conformal_qhat"] = qhat
    return out


def evaluate_conformal_sets(
    labels_df: pd.DataFrame,
    conformal_df: pd.DataFrame,
    thresholds_df: pd.DataFrame,
    image_column: str = "image",
) -> pd.DataFrame:
    """Evaluate empirical coverage and set-size behavior for conformal outputs."""

    merged = labels_df.merge(conformal_df, on=image_column, suffixes=("_true", "_pred"))
    rows: List[Dict[str, Any]] = []
    for _, threshold_row in thresholds_df.iterrows():
        label = str(threshold_row["label"])
        true_col = f"{label}_true" if f"{label}_true" in merged.columns else label
        set_col = f"{label}_conformal_set"
        size_col = f"{label}_conformal_set_size"
        neg_col = f"{label}_conformal_includes_negative"
        pos_col = f"{label}_conformal_includes_positive"
        if true_col not in merged.columns or set_col not in merged.columns:
            continue
        valid = merged[[true_col, set_col, size_col, neg_col, pos_col]].dropna()
        if valid.empty:
            continue
        y_true = valid[true_col].astype(int).to_numpy()
        includes_true = np.where(
            y_true == 1,
            valid[pos_col].astype(bool).to_numpy(),
            valid[neg_col].astype(bool).to_numpy(),
        )
        set_sizes = valid[size_col].astype(int).to_numpy()
        rows.append(
            {
                "label": label,
                "alpha": float(threshold_row["alpha"]),
                "target_coverage": float(threshold_row["target_coverage"]),
                "empirical_coverage": float(np.mean(includes_true)),
                "coverage_gap": float(np.mean(includes_true) - float(threshold_row["target_coverage"])),
                "n_eval": int(len(valid)),
                "mean_set_size": float(np.mean(set_sizes)),
                "singleton_rate": float(np.mean(set_sizes == 1)),
                "ambiguous_rate": float(np.mean(set_sizes == 2)),
                "empty_rate": float(np.mean(set_sizes == 0)),
                "qhat": float(threshold_row["qhat"]),
            }
        )
    return pd.DataFrame(rows)


def split_calibration_eval(
    df: pd.DataFrame,
    calibration_fraction: float = 0.5,
    seed: int = 13,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministically split a dataframe into calibration and evaluation rows."""

    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between 0 and 1")
    rng = np.random.default_rng(seed)
    indices = np.arange(len(df))
    rng.shuffle(indices)
    n_calibration = max(1, min(len(df) - 1, int(round(calibration_fraction * len(df)))))
    calibration_idx = indices[:n_calibration]
    eval_idx = indices[n_calibration:]
    return df.iloc[calibration_idx].reset_index(drop=True), df.iloc[eval_idx].reset_index(drop=True)


def write_conformal_report(
    output_path: str | Path,
    summary_df: pd.DataFrame,
    dataset_name: str,
    alpha: float,
    calibration_fraction: float | None = None,
) -> Path:
    """Write a Markdown report for conformal prediction-set results."""

    lines = [
        "# Conformal Prediction Report",
        "",
        f"Dataset: {dataset_name}",
        f"Alpha: `{alpha}`",
        f"Target coverage: `{1.0 - alpha:.3f}`",
    ]
    if calibration_fraction is not None:
        lines.append(f"Calibration fraction: `{calibration_fraction}`")
    lines.extend(
        [
            "",
            "## What this measures",
            "Split conformal prediction converts probabilities into per-label prediction sets. A set can contain `negative`, `positive`, both labels, or no label. Larger and ambiguous sets indicate cases that should be routed to human review rather than treated as confident automated outputs.",
            "",
            "## Summary",
        ]
    )
    if summary_df.empty:
        lines.append("No conformal summary was produced.")
    else:
        lines.extend(
            [
                "",
                "| Label | Target coverage | Empirical coverage | Mean set size | Singleton rate | Ambiguous rate | Empty rate |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in summary_df.iterrows():
            lines.append(
                f"| {row['label']} | {float(row['target_coverage']):.3f} | {float(row['empirical_coverage']):.3f} | {float(row['mean_set_size']):.3f} | {float(row['singleton_rate']):.3f} | {float(row['ambiguous_rate']):.3f} | {float(row['empty_rate']):.3f} |"
            )
    lines.extend(
        [
            "",
            "## Responsible-use note",
            "Conformal coverage is only meaningful under the calibration/evaluation distribution assumptions. This report is for research reliability analysis and does not establish clinical safety or regulatory compliance.",
            "",
        ]
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out

"""Selective prediction and abstention utilities for XRayMind."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score

from .metrics import specificity_score


def confidence_from_probability(y_score: np.ndarray) -> np.ndarray:
    """Return binary-classification confidence from pathology probability.

    Confidence is distance from the 0.5 decision boundary mapped to [0.5, 1.0].
    A score near 0 or 1 is treated as high confidence; a score near 0.5 is low confidence.
    """

    scores = np.asarray(y_score, dtype=float)
    return np.maximum(scores, 1.0 - scores)


def entropy_uncertainty(y_score: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Binary entropy uncertainty. Higher means more uncertain."""

    scores = np.clip(np.asarray(y_score, dtype=float), eps, 1.0 - eps)
    return -(scores * np.log(scores) + (1.0 - scores) * np.log(1.0 - scores))


def confidence_from_uncertainty(uncertainty: Sequence[float]) -> np.ndarray:
    """Convert an uncertainty vector into an ordering-compatible confidence score.

    Lower uncertainty should mean higher confidence. The returned score is min-max
    normalized to [0, 1], where 1 is most confident. Constant uncertainty receives
    a neutral confidence of 0.5 for every case.
    """

    values = np.asarray(uncertainty, dtype=float)
    if len(values) == 0:
        return values
    finite = np.isfinite(values)
    if not finite.any():
        return np.full_like(values, 0.5, dtype=float)
    clean = values.copy()
    clean[~finite] = np.nanmax(clean[finite])
    lo = float(np.min(clean))
    hi = float(np.max(clean))
    if np.isclose(lo, hi):
        return np.full_like(clean, 0.5, dtype=float)
    return 1.0 - ((clean - lo) / (hi - lo))


def combined_confidence(
    y_score: Sequence[float],
    uncertainty: Sequence[float] | None = None,
    uncertainty_weight: float = 0.5,
) -> np.ndarray:
    """Blend probability-margin confidence with optional uncertainty confidence.

    `uncertainty_weight=0` uses only probability distance from 0.5.
    `uncertainty_weight=1` uses only the supplied uncertainty ranking.
    """

    margin_conf = confidence_from_probability(np.asarray(y_score, dtype=float))
    if uncertainty is None:
        return margin_conf
    uncertainty_conf = confidence_from_uncertainty(uncertainty)
    w = float(np.clip(uncertainty_weight, 0.0, 1.0))
    return ((1.0 - w) * margin_conf) + (w * uncertainty_conf)


def selective_curve_for_label(
    y_true: Sequence[int],
    y_score: Sequence[float],
    label: str,
    threshold: float = 0.5,
    coverage_grid: Sequence[float] | None = None,
    confidence: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Compute metrics as progressively less-confident cases are deferred."""

    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)
    if len(y_true_arr) != len(y_score_arr):
        raise ValueError("y_true and y_score must have the same length")
    if len(y_true_arr) == 0:
        return pd.DataFrame()

    confidence_arr = np.asarray(confidence, dtype=float) if confidence is not None else confidence_from_probability(y_score_arr)
    if len(confidence_arr) != len(y_true_arr):
        raise ValueError("confidence must have the same length as y_true")
    order = np.argsort(-confidence_arr)
    grid = list(coverage_grid or np.round(np.linspace(0.1, 1.0, 10), 2))
    rows: List[Dict[str, float | int | str | None]] = []

    for coverage in grid:
        k = max(1, int(np.ceil(float(coverage) * len(y_true_arr))))
        selected = order[:k]
        selected_true = y_true_arr[selected]
        selected_score = y_score_arr[selected]
        selected_pred = (selected_score >= threshold).astype(int)
        has_both = len(set(selected_true.tolist())) == 2
        accuracy = float(np.mean(selected_pred == selected_true))
        risk = 1.0 - accuracy
        rows.append(
            {
                "label": label,
                "coverage": float(k / len(y_true_arr)),
                "defer_rate": float(1.0 - (k / len(y_true_arr))),
                "n_evaluated": int(k),
                "n_deferred": int(len(y_true_arr) - k),
                "mean_confidence": float(np.mean(confidence_arr[selected])),
                "min_confidence": float(np.min(confidence_arr[selected])),
                "selective_accuracy": accuracy,
                "selective_risk": risk,
                "auroc": float(roc_auc_score(selected_true, selected_score)) if has_both else None,
                "brier": float(brier_score_loss(selected_true, selected_score)),
                "sensitivity": float(recall_score(selected_true, selected_pred, zero_division=0)),
                "specificity": float(specificity_score(selected_true, selected_pred)),
                "precision": float(precision_score(selected_true, selected_pred, zero_division=0)),
                "f1": float(f1_score(selected_true, selected_pred, zero_division=0)),
            }
        )
    return pd.DataFrame(rows)


def evaluate_selective_predictions(
    labels_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    image_column: str = "image",
    labels: Sequence[str] | None = None,
    threshold: float = 0.5,
    coverage_grid: Sequence[float] | None = None,
    confidence_suffix: str = "_confidence",
    uncertainty_suffix: str | None = None,
    uncertainty_weight: float = 0.5,
) -> pd.DataFrame:
    """Evaluate selective prediction curves for all labels.

    If `<label>_confidence` exists in `pred_df`, it is used to rank cases for
    deferral. Otherwise, if `uncertainty_suffix` is provided and the matching
    column exists, the uncertainty is blended with probability-margin confidence.
    Otherwise, probability distance from 0.5 is used.
    """

    merged = labels_df.merge(pred_df, on=image_column, suffixes=("_true", "_pred"))
    label_names = list(labels or [c for c in labels_df.columns if c != image_column])
    frames: List[pd.DataFrame] = []
    for label in label_names:
        true_col = f"{label}_true" if f"{label}_true" in merged.columns else label
        pred_col = f"{label}_pred" if f"{label}_pred" in merged.columns else label
        conf_col = f"{label}{confidence_suffix}"
        uncertainty_col = f"{label}{uncertainty_suffix}" if uncertainty_suffix else None
        if true_col not in merged.columns or pred_col not in merged.columns:
            continue
        y_true = merged[true_col].astype(int).to_numpy()
        y_score = merged[pred_col].astype(float).to_numpy()
        if len(set(y_true.tolist())) < 2:
            continue
        confidence = None
        if conf_col in merged.columns:
            confidence = merged[conf_col].astype(float).to_numpy()
        elif uncertainty_col and uncertainty_col in merged.columns:
            confidence = combined_confidence(
                y_score=y_score,
                uncertainty=merged[uncertainty_col].astype(float).to_numpy(),
                uncertainty_weight=uncertainty_weight,
            )
        frames.append(
            selective_curve_for_label(
                y_true=y_true,
                y_score=y_score,
                label=label,
                threshold=threshold,
                coverage_grid=coverage_grid,
                confidence=confidence,
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_selective_curves(curves_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate selective curves across labels by coverage."""

    if curves_df.empty:
        return pd.DataFrame()
    summary = (
        curves_df.groupby("coverage")
        .agg(
            labels_evaluated=("label", "count"),
            mean_selective_accuracy=("selective_accuracy", "mean"),
            mean_selective_risk=("selective_risk", "mean"),
            mean_auroc=("auroc", "mean"),
            mean_brier=("brier", "mean"),
            mean_f1=("f1", "mean"),
            mean_confidence=("mean_confidence", "mean"),
            mean_defer_rate=("defer_rate", "mean"),
        )
        .reset_index()
        .sort_values("coverage")
    )
    return summary


def choose_operating_points(summary_df: pd.DataFrame, max_risk: float | None = None, min_coverage: float | None = None) -> Dict[str, float | int | None]:
    """Choose a simple abstention operating point from aggregate curves."""

    if summary_df.empty:
        return {"coverage": None, "defer_rate": None, "mean_selective_risk": None, "reason": "no_valid_curves"}
    candidates = summary_df.copy()
    if max_risk is not None:
        candidates = candidates[candidates["mean_selective_risk"] <= max_risk]
    if min_coverage is not None:
        candidates = candidates[candidates["coverage"] >= min_coverage]
    if candidates.empty:
        row = summary_df.sort_values("mean_selective_risk", ascending=True).iloc[0]
        reason = "no_point_met_constraints; selected_lowest_risk"
    else:
        row = candidates.sort_values(["coverage", "mean_selective_risk"], ascending=[False, True]).iloc[0]
        reason = "met_constraints"
    return {
        "coverage": float(row["coverage"]),
        "defer_rate": float(row["mean_defer_rate"]),
        "mean_selective_risk": float(row["mean_selective_risk"]),
        "mean_selective_accuracy": float(row["mean_selective_accuracy"]),
        "mean_auroc": float(row["mean_auroc"]) if not pd.isna(row["mean_auroc"]) else None,
        "labels_evaluated": int(row["labels_evaluated"]),
        "reason": reason,
    }


def write_selective_report(
    output_path: str | Path,
    summary_df: pd.DataFrame,
    operating_point: Dict[str, float | int | str | None],
    dataset_name: str,
    model_name: str,
    plot_path: str | Path | None = None,
    confidence_method: str = "probability_margin",
) -> Path:
    """Write a Markdown report for selective prediction results."""

    lines = [
        f"# Selective Prediction Report: {model_name}",
        "",
        f"Dataset: {dataset_name}",
        f"Confidence method: `{confidence_method}`",
        "",
        "## What this measures",
        "Selective prediction evaluates performance after deferring the least-confident cases. This approximates a human-in-the-loop workflow where uncertain cases are escalated instead of receiving an automatic prediction.",
        "",
        "## Recommended operating point",
    ]
    for key, value in operating_point.items():
        lines.append(f"- `{key}`: {value}")
    if plot_path:
        lines.extend(["", "## Curve", f"![Selective risk curve]({plot_path})"])
    if not summary_df.empty:
        lines.extend(["", "## Aggregate curve", "", "| Coverage | Defer rate | Mean risk | Mean accuracy | Mean AUROC |", "|---:|---:|---:|---:|---:|"])
        for _, row in summary_df.iterrows():
            mean_auroc = row.get("mean_auroc")
            auroc_text = "NA" if pd.isna(mean_auroc) else f"{float(mean_auroc):.4f}"
            lines.append(
                f"| {float(row['coverage']):.2f} | {float(row['mean_defer_rate']):.2f} | {float(row['mean_selective_risk']):.4f} | {float(row['mean_selective_accuracy']):.4f} | {auroc_text} |"
            )
    lines.extend(
        [
            "",
            "## Responsible-use note",
            "This is a research-only deferral analysis. It does not validate clinical safety, deployment readiness, or regulatory compliance.",
            "",
        ]
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out

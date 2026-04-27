"""Ensemble prediction and uncertainty utilities for XRayMind."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from .config import DEFAULT_MODEL_NAME
from .evaluation import run_predictions
from .selective import combined_confidence, entropy_uncertainty


def ensemble_from_prediction_frames(
    prediction_frames: Dict[str, pd.DataFrame],
    image_column: str = "image",
    labels: Sequence[str] | None = None,
    uncertainty_method: str = "std",
    confidence_uncertainty_weight: float = 0.5,
) -> pd.DataFrame:
    """Aggregate model prediction frames into ensemble mean and uncertainty columns.

    Output columns keep the standard XRayMind convention where each label column
    stores the probability used by existing metrics. Additional columns are added:

    - `<label>_ensemble_std`: standard deviation across model probabilities.
    - `<label>_ensemble_range`: max minus min model probability.
    - `<label>_ensemble_entropy`: binary entropy of the ensemble mean.
    - `<label>_ensemble_uncertainty`: selected uncertainty score.
    - `<label>_confidence`: dataset-level confidence score used by selective prediction.
    - `<label>__<model>`: raw probability from each member model.
    """

    if not prediction_frames:
        raise ValueError("prediction_frames must contain at least one model")

    model_names = list(prediction_frames.keys())
    merged: pd.DataFrame | None = None
    for model_name, frame in prediction_frames.items():
        model_frame = frame.copy()
        candidate_labels = [c for c in model_frame.columns if c != image_column]
        rename_map = {label: f"{label}__{model_name}" for label in candidate_labels}
        model_frame = model_frame.rename(columns=rename_map)
        merged = model_frame if merged is None else merged.merge(model_frame, on=image_column, how="inner")

    if merged is None or merged.empty:
        return pd.DataFrame(columns=[image_column])

    if labels is None:
        label_set = set()
        for frame in prediction_frames.values():
            label_set.update([c for c in frame.columns if c != image_column])
        labels = sorted(label_set)

    records: List[Dict[str, Any]] = []
    for _, row in merged.iterrows():
        out: Dict[str, Any] = {image_column: row[image_column]}
        for label in labels:
            member_cols = [f"{label}__{model}" for model in model_names if f"{label}__{model}" in merged.columns]
            if not member_cols:
                continue
            values = row[member_cols].astype(float).to_numpy()
            mean_prob = float(np.mean(values))
            std_prob = float(np.std(values, ddof=0))
            range_prob = float(np.max(values) - np.min(values))
            entropy = float(entropy_uncertainty(np.asarray([mean_prob]))[0])
            if uncertainty_method == "range":
                uncertainty = range_prob
            elif uncertainty_method == "entropy":
                uncertainty = entropy
            elif uncertainty_method == "std_entropy":
                uncertainty = std_prob + entropy
            else:
                uncertainty = std_prob
            out[label] = mean_prob
            out[f"{label}_ensemble_std"] = std_prob
            out[f"{label}_ensemble_range"] = range_prob
            out[f"{label}_ensemble_entropy"] = entropy
            out[f"{label}_ensemble_uncertainty"] = float(uncertainty)
            for col in member_cols:
                out[col] = float(row[col])
        records.append(out)

    ensemble = pd.DataFrame(records)

    # Important: confidence-from-uncertainty is a dataset-level ranking. Computing it
    # one row at a time collapses min-max normalization to a neutral value and makes
    # uncertainty-aware selective prediction weaker than intended.
    for label in labels:
        uncertainty_col = f"{label}_ensemble_uncertainty"
        if label in ensemble.columns and uncertainty_col in ensemble.columns:
            ensemble[f"{label}_confidence"] = combined_confidence(
                y_score=ensemble[label].astype(float).to_numpy(),
                uncertainty=ensemble[uncertainty_col].astype(float).to_numpy(),
                uncertainty_weight=confidence_uncertainty_weight,
            )
    return ensemble


def run_ensemble_predictions(
    image_dir: str | Path,
    labels_df: pd.DataFrame,
    models: Sequence[str] | None = None,
    image_column: str = "image",
    labels: Sequence[str] | None = None,
    limit: int | None = None,
    top_k: int = 100,
    threshold: float = 0.5,
    uncertainty_method: str = "std",
    confidence_uncertainty_weight: float = 0.5,
) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Run multiple TorchXRayVision models and return ensemble predictions."""

    selected_models = list(models or [DEFAULT_MODEL_NAME])
    member_frames: Dict[str, pd.DataFrame] = {}
    for model_name in selected_models:
        member_frames[model_name] = run_predictions(
            image_dir=image_dir,
            labels_df=labels_df,
            image_column=image_column,
            model_name=model_name,
            limit=limit,
            top_k=top_k,
            threshold=threshold,
        )
    ensemble = ensemble_from_prediction_frames(
        member_frames,
        image_column=image_column,
        labels=labels,
        uncertainty_method=uncertainty_method,
        confidence_uncertainty_weight=confidence_uncertainty_weight,
    )
    return ensemble, member_frames


def summarize_ensemble_uncertainty(
    ensemble_df: pd.DataFrame,
    image_column: str = "image",
    labels: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Summarize ensemble uncertainty columns per label."""

    if ensemble_df.empty:
        return pd.DataFrame()
    if labels is None:
        labels = sorted(
            c.replace("_ensemble_uncertainty", "")
            for c in ensemble_df.columns
            if c.endswith("_ensemble_uncertainty")
        )
    rows: List[Dict[str, Any]] = []
    for label in labels:
        uncertainty_col = f"{label}_ensemble_uncertainty"
        std_col = f"{label}_ensemble_std"
        range_col = f"{label}_ensemble_range"
        entropy_col = f"{label}_ensemble_entropy"
        confidence_col = f"{label}_confidence"
        if uncertainty_col not in ensemble_df.columns:
            continue
        row: Dict[str, Any] = {"label": label, "n": int(len(ensemble_df))}
        for name, col in {
            "mean_uncertainty": uncertainty_col,
            "mean_std": std_col,
            "mean_range": range_col,
            "mean_entropy": entropy_col,
            "mean_confidence": confidence_col,
        }.items():
            if col in ensemble_df.columns:
                row[name] = float(ensemble_df[col].astype(float).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def write_ensemble_report(
    output_path: str | Path,
    model_names: Sequence[str],
    uncertainty_summary: pd.DataFrame,
    dataset_name: str,
    uncertainty_method: str = "std",
) -> Path:
    """Write a Markdown report describing ensemble uncertainty outputs."""

    lines = [
        "# Ensemble Uncertainty Report",
        "",
        f"Dataset: {dataset_name}",
        f"Models: {', '.join(model_names)}",
        f"Uncertainty method: `{uncertainty_method}`",
        "",
        "## What this measures",
        "The ensemble prediction is the mean probability across member models. Uncertainty is derived from disagreement across model probabilities and can be used for selective prediction or human review routing.",
        "",
        "## Per-label uncertainty summary",
    ]
    if uncertainty_summary.empty:
        lines.append("No uncertainty summary was produced.")
    else:
        lines.extend(["", "| Label | N | Mean uncertainty | Mean std | Mean range | Mean entropy | Mean confidence |", "|---|---:|---:|---:|---:|---:|---:|"])
        for _, row in uncertainty_summary.iterrows():
            lines.append(
                f"| {row.get('label')} | {int(row.get('n', 0))} | {float(row.get('mean_uncertainty', float('nan'))):.4f} | {float(row.get('mean_std', float('nan'))):.4f} | {float(row.get('mean_range', float('nan'))):.4f} | {float(row.get('mean_entropy', float('nan'))):.4f} | {float(row.get('mean_confidence', float('nan'))):.4f} |"
            )
    lines.extend(
        [
            "",
            "## Responsible-use note",
            "Ensemble disagreement is a research uncertainty signal. It is not a calibrated clinical confidence score and does not establish deployment safety.",
            "",
        ]
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out

"""Reusable dataset evaluation utilities for XRayMind."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .bootstrap import bootstrap_ci
from .config import DEFAULT_MODEL_NAME, DISCLAIMER
from .inference import predict_image
from .metrics import compute_binary_metrics, tune_threshold
from .plots import save_reliability_diagram


def label_columns(labels_df: pd.DataFrame, image_column: str, subgroup_columns: Sequence[str] | None = None) -> List[str]:
    """Return candidate pathology label columns."""

    excluded = {image_column, *(subgroup_columns or [])}
    return [column for column in labels_df.columns if column not in excluded]


def run_predictions(
    image_dir: str | Path,
    labels_df: pd.DataFrame,
    image_column: str = "image",
    model_name: str = DEFAULT_MODEL_NAME,
    limit: Optional[int] = None,
    top_k: int = 100,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Run model predictions for every row in a label CSV."""

    image_root = Path(image_dir)
    work_df = labels_df.head(limit) if limit else labels_df
    records: List[Dict[str, Any]] = []
    for _, row in work_df.iterrows():
        image_path = image_root / str(row[image_column])
        prediction = predict_image(image_path, model_name=model_name, top_k=top_k, threshold=threshold)
        probs = {p["label"]: p["probability"] for p in prediction["predictions"]}
        records.append({image_column: row[image_column], **probs})
    return pd.DataFrame(records)


def evaluate_predictions(
    labels_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    image_column: str = "image",
    labels: Sequence[str] | None = None,
    threshold: float = 0.5,
    tune_thresholds: bool = False,
    threshold_objective: str = "f1",
    bootstrap: int = 0,
    n_bins: int = 10,
    plot_dir: str | Path | None = None,
    plot_prefix: str = "",
) -> pd.DataFrame:
    """Compute per-label metrics from label and prediction dataframes."""

    merged = labels_df.merge(pred_df, on=image_column, suffixes=("_true", "_pred"))
    metric_rows: List[Dict[str, Any]] = []
    candidate_labels = list(labels or [c for c in labels_df.columns if c != image_column])

    for label in candidate_labels:
        pred_col = label if label in merged.columns else f"{label}_pred"
        true_col = label if label in labels_df.columns and label in merged.columns else f"{label}_true"
        if pred_col not in merged.columns or true_col not in merged.columns:
            continue
        y_true = merged[true_col].astype(int).to_numpy()
        y_score = merged[pred_col].astype(float).to_numpy()
        if len(set(y_true.tolist())) < 2:
            continue

        chosen_threshold = (
            tune_threshold(y_true, y_score, objective=threshold_objective)
            if tune_thresholds
            else threshold
        )
        row = compute_binary_metrics(
            label=label,
            y_true=y_true,
            y_score=y_score,
            threshold=chosen_threshold,
            n_bins=n_bins,
        ).as_dict()

        if bootstrap > 0:
            auroc_low, auroc_high = bootstrap_ci(y_true, y_score, roc_auc_score, n_bootstrap=bootstrap)
            auprc_low, auprc_high = bootstrap_ci(y_true, y_score, average_precision_score, n_bootstrap=bootstrap)
            row.update(
                {
                    "auroc_ci_low": auroc_low,
                    "auroc_ci_high": auroc_high,
                    "auprc_ci_low": auprc_low,
                    "auprc_ci_high": auprc_high,
                }
            )

        if plot_dir:
            safe_label = label.replace("/", "_").replace(" ", "_")
            plot_path = Path(plot_dir) / f"{plot_prefix}{safe_label}_reliability.png"
            save_reliability_diagram(
                y_true,
                y_score,
                plot_path,
                n_bins=n_bins,
                title=f"{plot_prefix}{label} reliability",
            )
            row["reliability_plot"] = str(plot_path)
        metric_rows.append(row)
    return pd.DataFrame(metric_rows)


def evaluate_subgroups(
    labels_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    subgroup_columns: Sequence[str],
    image_column: str = "image",
    labels: Sequence[str] | None = None,
    min_group_size: int = 10,
    threshold: float = 0.5,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute per-label metrics for each subgroup value."""

    rows: List[pd.DataFrame] = []
    merged = labels_df.merge(pred_df, on=image_column, suffixes=("_true", "_pred"))
    for subgroup_col in subgroup_columns:
        if subgroup_col not in merged.columns:
            continue
        for subgroup_value, group in merged.groupby(subgroup_col):
            if len(group) < min_group_size:
                continue
            label_cols = list(labels or [c for c in labels_df.columns if c not in {image_column, *subgroup_columns}])
            group_labels = group[[image_column, *[f"{c}_true" if f"{c}_true" in group.columns else c for c in label_cols]]].copy()
            rename_map = {f"{c}_true": c for c in label_cols if f"{c}_true" in group_labels.columns}
            group_labels = group_labels.rename(columns=rename_map)
            group_preds = group[[image_column, *[f"{c}_pred" if f"{c}_pred" in group.columns else c for c in label_cols if (f"{c}_pred" in group.columns or c in group.columns)]]].copy()
            rename_map = {f"{c}_pred": c for c in label_cols if f"{c}_pred" in group_preds.columns}
            group_preds = group_preds.rename(columns=rename_map)
            metrics = evaluate_predictions(
                group_labels,
                group_preds,
                image_column=image_column,
                labels=label_cols,
                threshold=threshold,
                n_bins=n_bins,
            )
            if not metrics.empty:
                metrics.insert(0, "subgroup_value", subgroup_value)
                metrics.insert(0, "subgroup_column", subgroup_col)
                metrics.insert(0, "subgroup_n", len(group))
                rows.append(metrics)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def write_dataset_card(
    labels_df: pd.DataFrame,
    output_path: str | Path,
    image_column: str = "image",
    subgroup_columns: Sequence[str] | None = None,
    dataset_name: str = "XRayMind folder dataset",
) -> Path:
    """Write a lightweight Markdown dataset card from a labels CSV."""

    labels = label_columns(labels_df, image_column, subgroup_columns)
    lines = [
        f"# Dataset Card: {dataset_name}",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Intended use",
        "Research benchmarking for chest X-ray classification models. Not for clinical diagnosis.",
        "",
        "## Size",
        f"Number of rows: {len(labels_df)}",
        f"Image column: `{image_column}`",
        "",
        "## Labels",
    ]
    for label in labels:
        prevalence = float(labels_df[label].astype(float).mean()) if label in labels_df else float("nan")
        lines.append(f"- `{label}` prevalence: {prevalence:.4f}")
    if subgroup_columns:
        lines.extend(["", "## Subgroups"])
        for col in subgroup_columns:
            if col in labels_df.columns:
                counts = labels_df[col].value_counts(dropna=False).head(20)
                lines.append(f"### {col}")
                for value, count in counts.items():
                    lines.append(f"- `{value}`: {int(count)}")
    lines.extend(["", "## Responsible-use note", DISCLAIMER, ""])
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_benchmark_model_card(
    metrics_df: pd.DataFrame,
    output_path: str | Path,
    model_name: str,
    dataset_name: str,
    subgroup_metrics_path: str | Path | None = None,
) -> Path:
    """Write a model card summary for one benchmark run."""

    lines = [
        f"# Model Card: {model_name}",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Model",
        f"- Model name: `{model_name}`",
        "- Source: TorchXRayVision pretrained model registry",
        "",
        "## Evaluation dataset",
        f"- Dataset: {dataset_name}",
        f"- Labels evaluated: {len(metrics_df) if metrics_df is not None else 0}",
        "",
        "## Aggregate metrics",
    ]
    if metrics_df is not None and not metrics_df.empty:
        aggregate_cols = ["auroc", "auprc", "brier", "ece", "sensitivity", "specificity", "precision", "f1"]
        for col in aggregate_cols:
            if col in metrics_df.columns:
                lines.append(f"- Mean {col}: {float(metrics_df[col].dropna().mean()):.4f}")
        lines.extend(["", "## Per-label metrics", "", "| Label | AUROC | AUPRC | Brier | ECE | F1 |", "|---|---:|---:|---:|---:|---:|"])
        for _, row in metrics_df.iterrows():
            lines.append(
                f"| {row.get('label')} | {row.get('auroc', float('nan')):.4f} | {row.get('auprc', float('nan')):.4f} | {row.get('brier', float('nan')):.4f} | {row.get('ece', float('nan')):.4f} | {row.get('f1', float('nan')):.4f} |"
            )
    else:
        lines.append("No valid per-label metrics were produced. Check label names and class balance.")
    if subgroup_metrics_path:
        lines.extend(["", "## Subgroup evaluation", f"Subgroup metrics were written to `{subgroup_metrics_path}`."])
    lines.extend(
        [
            "",
            "## Intended use",
            "Research benchmarking, educational demos, reliability analysis, and model comparison. Not for clinical deployment.",
            "",
            "## Limitations",
            "- Scores depend on the dataset, preprocessing, label definitions, and domain shift.",
            "- Calibration and thresholds must be validated on the target setting.",
            "- Heatmaps are sensitivity visualizations, not confirmed lesion localization.",
            "- This card does not establish regulatory, clinical, or safety validation.",
            "",
            "## Responsible-use note",
            DISCLAIMER,
            "",
        ]
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_run_manifest(output_path: str | Path, **payload: Any) -> Path:
    """Write a JSON manifest for a benchmark run."""

    manifest = {
        "schema_version": "xraymind.benchmark_run.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "disclaimer": DISCLAIMER,
        **payload,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return out

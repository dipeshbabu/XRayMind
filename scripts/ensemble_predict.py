"""Run multi-model ensemble prediction, uncertainty, evaluation, and abstention.

Example:
python scripts/ensemble_predict.py \
  --image-dir data/images \
  --labels data/labels.csv \
  --models densenet121-res224-all resnet50-res512-all \
  --out-dir outputs/ensemble_v0_8 \
  --selective \
  --max-risk 0.15
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from xraymind.ensemble import (
    run_ensemble_predictions,
    summarize_ensemble_uncertainty,
    write_ensemble_report,
)
from xraymind.evaluation import evaluate_predictions, label_columns, write_run_manifest
from xraymind.selective import (
    choose_operating_points,
    evaluate_selective_predictions,
    summarize_selective_curves,
    write_selective_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run XRayMind ensemble uncertainty evaluation")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out-dir", default="outputs/ensemble_v0_8")
    parser.add_argument("--models", nargs="+", required=True, help="Two or more TorchXRayVision model names")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--labels-to-evaluate", nargs="*", default=None)
    parser.add_argument("--subgroups", nargs="*", default=[])
    parser.add_argument("--dataset", default="XRayMind folder dataset")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--uncertainty-method",
        choices=["std", "range", "entropy", "std_entropy"],
        default="std",
    )
    parser.add_argument("--confidence-uncertainty-weight", type=float, default=0.5)
    parser.add_argument("--selective", action="store_true", help="Also generate ensemble selective prediction artifacts")
    parser.add_argument("--coverage-grid", nargs="*", type=float, default=None)
    parser.add_argument("--max-risk", type=float, default=None)
    parser.add_argument("--min-coverage", type=float, default=None)
    return parser.parse_args()


def save_selective_curve_plot(summary_df: pd.DataFrame, out_path: Path) -> Path | None:
    if summary_df.empty:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6, 4))
    plt.plot(summary_df["coverage"], summary_df["mean_selective_risk"], marker="o")
    plt.xlabel("Coverage: fraction of cases automatically predicted")
    plt.ylabel("Selective risk: 1 - accuracy")
    plt.title("Ensemble selective risk curve")
    plt.xlim(0, 1.05)
    plt.ylim(bottom=0)
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main() -> int:
    args = parse_args()
    if len(args.models) < 2:
        raise ValueError("Ensemble uncertainty is most useful with at least two models")

    out_dir = Path(args.out_dir)
    members_dir = out_dir / "members"
    out_dir.mkdir(parents=True, exist_ok=True)
    members_dir.mkdir(parents=True, exist_ok=True)

    labels_df = pd.read_csv(args.labels)
    label_names = args.labels_to_evaluate or label_columns(labels_df, args.image_column, args.subgroups)

    ensemble_df, member_frames = run_ensemble_predictions(
        image_dir=args.image_dir,
        labels_df=labels_df,
        models=args.models,
        image_column=args.image_column,
        labels=label_names,
        limit=args.limit,
        top_k=args.top_k,
        threshold=args.threshold,
        uncertainty_method=args.uncertainty_method,
        confidence_uncertainty_weight=args.confidence_uncertainty_weight,
    )

    for model_name, frame in member_frames.items():
        safe_name = model_name.replace("/", "_").replace(" ", "_")
        frame.to_csv(members_dir / f"{safe_name}_predictions.csv", index=False)

    predictions_path = out_dir / "ensemble_predictions.csv"
    ensemble_df.to_csv(predictions_path, index=False)

    metrics_df = evaluate_predictions(
        labels_df=labels_df,
        pred_df=ensemble_df,
        image_column=args.image_column,
        labels=label_names,
        threshold=args.threshold,
    )
    metrics_path = out_dir / "ensemble_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    uncertainty_summary = summarize_ensemble_uncertainty(
        ensemble_df=ensemble_df,
        image_column=args.image_column,
        labels=label_names,
    )
    uncertainty_path = out_dir / "ensemble_uncertainty_summary.csv"
    uncertainty_summary.to_csv(uncertainty_path, index=False)

    ensemble_report_path = write_ensemble_report(
        output_path=out_dir / "ENSEMBLE_UNCERTAINTY_REPORT.md",
        model_names=args.models,
        uncertainty_summary=uncertainty_summary,
        dataset_name=args.dataset,
        uncertainty_method=args.uncertainty_method,
    )

    selective_payload = None
    if args.selective:
        selective_dir = out_dir / "selective"
        selective_dir.mkdir(parents=True, exist_ok=True)
        curves_df = evaluate_selective_predictions(
            labels_df=labels_df,
            pred_df=ensemble_df,
            image_column=args.image_column,
            labels=label_names,
            threshold=args.threshold,
            coverage_grid=args.coverage_grid,
            confidence_suffix="_confidence",
            uncertainty_suffix="_ensemble_uncertainty",
            uncertainty_weight=args.confidence_uncertainty_weight,
        )
        curves_path = selective_dir / "selective_curves.csv"
        curves_df.to_csv(curves_path, index=False)
        summary_df = summarize_selective_curves(curves_df)
        summary_path = selective_dir / "selective_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        operating_point = choose_operating_points(summary_df, max_risk=args.max_risk, min_coverage=args.min_coverage)
        operating_path = selective_dir / "operating_point.json"
        operating_path.write_text(json.dumps(operating_point, indent=2), encoding="utf-8")
        plot_path = save_selective_curve_plot(summary_df, selective_dir / "selective_risk_curve.png")
        report_path = write_selective_report(
            output_path=selective_dir / "SELECTIVE_PREDICTION_REPORT.md",
            summary_df=summary_df,
            operating_point=operating_point,
            dataset_name=args.dataset,
            model_name="ensemble(" + ", ".join(args.models) + ")",
            plot_path=plot_path.name if plot_path else None,
            confidence_method="ensemble_uncertainty",
        )
        selective_payload = {
            "curves_csv": str(curves_path),
            "summary_csv": str(summary_path),
            "operating_point_json": str(operating_path),
            "report": str(report_path),
            "plot": str(plot_path) if plot_path else None,
        }

    manifest_path = write_run_manifest(
        out_dir / "run_manifest.json",
        run_type="ensemble_uncertainty",
        models=args.models,
        dataset=args.dataset,
        labels_csv=args.labels,
        image_dir=args.image_dir,
        labels_evaluated=label_names,
        limit=args.limit,
        threshold=args.threshold,
        uncertainty_method=args.uncertainty_method,
        confidence_uncertainty_weight=args.confidence_uncertainty_weight,
        predictions_csv=str(predictions_path),
        metrics_csv=str(metrics_path),
        uncertainty_summary_csv=str(uncertainty_path),
        ensemble_report=str(ensemble_report_path),
        selective=selective_payload,
    )

    print(f"Saved ensemble predictions to {predictions_path}")
    print(f"Saved ensemble metrics to {metrics_path}")
    print(f"Saved uncertainty summary to {uncertainty_path}")
    print(f"Saved ensemble report to {ensemble_report_path}")
    if selective_payload:
        print(f"Saved selective artifacts to {out_dir / 'selective'}")
    print(f"Saved run manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

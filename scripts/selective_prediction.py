"""Generate selective prediction / abstention artifacts from labels and predictions.

Example:
python scripts/selective_prediction.py \
  --labels data/labels.csv \
  --predictions outputs/benchmark_v0_6/densenet121-res224-all/predictions.csv \
  --model densenet121-res224-all \
  --dataset "NIH validation split" \
  --out-dir outputs/selective_v0_7 \
  --max-risk 0.15

For ensemble prediction CSVs produced by scripts/ensemble_predict.py, use:
python scripts/selective_prediction.py \
  --labels data/labels.csv \
  --predictions outputs/ensemble_v0_8/ensemble_predictions.csv \
  --confidence-method ensemble_uncertainty \
  --uncertainty-suffix _ensemble_uncertainty
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from xraymind.evaluation import label_columns
from xraymind.selective import (
    choose_operating_points,
    evaluate_selective_predictions,
    summarize_selective_curves,
    write_selective_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate selective prediction and abstention curves")
    parser.add_argument("--labels", required=True, help="Ground-truth CSV")
    parser.add_argument("--predictions", required=True, help="Prediction CSV produced by XRayMind")
    parser.add_argument("--out-dir", default="outputs/selective_v0_7")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--labels-to-evaluate", nargs="*", default=None)
    parser.add_argument("--subgroups", nargs="*", default=[])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--coverage-grid", nargs="*", type=float, default=None)
    parser.add_argument("--max-risk", type=float, default=None)
    parser.add_argument("--min-coverage", type=float, default=None)
    parser.add_argument("--model", default="xraymind-model")
    parser.add_argument("--dataset", default="XRayMind evaluation dataset")
    parser.add_argument(
        "--confidence-method",
        choices=["probability_margin", "prediction_confidence_column", "ensemble_uncertainty"],
        default="probability_margin",
        help="How to rank cases for abstention. Ensemble CSVs should use ensemble_uncertainty.",
    )
    parser.add_argument("--confidence-suffix", default="_confidence")
    parser.add_argument("--uncertainty-suffix", default=None)
    parser.add_argument("--uncertainty-weight", type=float, default=0.5)
    return parser.parse_args()


def save_curve_plot(summary_df: pd.DataFrame, out_path: Path) -> Path | None:
    if summary_df.empty:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6, 4))
    plt.plot(summary_df["coverage"], summary_df["mean_selective_risk"], marker="o", label="Mean selective risk")
    plt.xlabel("Coverage: fraction of cases automatically predicted")
    plt.ylabel("Selective risk: 1 - accuracy")
    plt.title("Selective prediction risk curve")
    plt.xlim(0, 1.05)
    plt.ylim(bottom=0)
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_df = pd.read_csv(args.labels)
    pred_df = pd.read_csv(args.predictions)
    label_names = args.labels_to_evaluate or label_columns(labels_df, args.image_column, args.subgroups)

    uncertainty_suffix = args.uncertainty_suffix
    if args.confidence_method == "ensemble_uncertainty" and uncertainty_suffix is None:
        uncertainty_suffix = "_ensemble_uncertainty"

    curves_df = evaluate_selective_predictions(
        labels_df=labels_df,
        pred_df=pred_df,
        image_column=args.image_column,
        labels=label_names,
        threshold=args.threshold,
        coverage_grid=args.coverage_grid,
        confidence_suffix=args.confidence_suffix,
        uncertainty_suffix=uncertainty_suffix,
        uncertainty_weight=args.uncertainty_weight,
    )
    curves_path = out_dir / "selective_curves.csv"
    curves_df.to_csv(curves_path, index=False)

    summary_df = summarize_selective_curves(curves_df)
    summary_path = out_dir / "selective_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    operating_point = choose_operating_points(
        summary_df,
        max_risk=args.max_risk,
        min_coverage=args.min_coverage,
    )
    operating_path = out_dir / "operating_point.json"
    operating_path.write_text(json.dumps(operating_point, indent=2), encoding="utf-8")

    plot_path = save_curve_plot(summary_df, out_dir / "selective_risk_curve.png")
    report_path = write_selective_report(
        output_path=out_dir / "SELECTIVE_PREDICTION_REPORT.md",
        summary_df=summary_df,
        operating_point=operating_point,
        dataset_name=args.dataset,
        model_name=args.model,
        plot_path=plot_path.name if plot_path else None,
        confidence_method=args.confidence_method,
    )

    print(f"Saved selective curves to {curves_path}")
    print(f"Saved selective summary to {summary_path}")
    print(f"Saved operating point to {operating_path}")
    if plot_path:
        print(f"Saved selective risk curve to {plot_path}")
    print(f"Saved report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Create conformal prediction sets from XRayMind probability outputs.

Example:
python scripts/conformal_predict.py \
  --labels data/labels.csv \
  --predictions outputs/benchmark_v0_7/model/predictions.csv \
  --out-dir outputs/conformal_v0_9 \
  --alpha 0.1 \
  --calibration-fraction 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from xraymind.conformal import (
    apply_conformal_sets,
    calibrate_conformal_thresholds,
    evaluate_conformal_sets,
    split_calibration_eval,
    write_conformal_report,
)
from xraymind.evaluation import label_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate conformal prediction sets for XRayMind predictions")
    parser.add_argument("--labels", required=True, help="CSV containing image names and binary label columns")
    parser.add_argument("--predictions", required=True, help="CSV containing image names and probability columns")
    parser.add_argument("--out-dir", default="outputs/conformal_v0_9")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--labels-list", nargs="*", default=None, help="Optional explicit pathology labels")
    parser.add_argument("--alpha", type=float, default=0.1, help="Miscoverage rate; 0.1 targets 90 percent coverage")
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dataset-name", default="XRayMind folder dataset")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_df = pd.read_csv(args.labels)
    pred_df = pd.read_csv(args.predictions)
    labels = args.labels_list or label_columns(labels_df, args.image_column, subgroup_columns=[])
    labels = [label for label in labels if label in pred_df.columns]

    calibration_labels, eval_labels = split_calibration_eval(
        labels_df,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
    )
    calibration_predictions = pred_df[pred_df[args.image_column].isin(calibration_labels[args.image_column])].reset_index(drop=True)
    eval_predictions = pred_df[pred_df[args.image_column].isin(eval_labels[args.image_column])].reset_index(drop=True)

    thresholds = calibrate_conformal_thresholds(
        labels_df=calibration_labels,
        pred_df=calibration_predictions,
        image_column=args.image_column,
        labels=labels,
        alpha=args.alpha,
    )
    thresholds_path = out_dir / "conformal_thresholds.csv"
    thresholds.to_csv(thresholds_path, index=False)

    conformal_eval = apply_conformal_sets(
        pred_df=eval_predictions,
        thresholds_df=thresholds,
        image_column=args.image_column,
    )
    conformal_eval_path = out_dir / "conformal_predictions.csv"
    conformal_eval.to_csv(conformal_eval_path, index=False)

    summary = evaluate_conformal_sets(
        labels_df=eval_labels,
        conformal_df=conformal_eval,
        thresholds_df=thresholds,
        image_column=args.image_column,
    )
    summary_path = out_dir / "conformal_summary.csv"
    summary.to_csv(summary_path, index=False)

    report_path = write_conformal_report(
        output_path=out_dir / "CONFORMAL_REPORT.md",
        summary_df=summary,
        dataset_name=args.dataset_name,
        alpha=args.alpha,
        calibration_fraction=args.calibration_fraction,
    )

    print(f"Saved conformal thresholds to {thresholds_path}")
    print(f"Saved conformal predictions to {conformal_eval_path}")
    print(f"Saved conformal summary to {summary_path}")
    print(f"Saved conformal report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

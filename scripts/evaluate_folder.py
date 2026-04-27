"""Evaluate XRayMind predictions over a labeled image folder.

Expected labels CSV format:
- an image filename column, default: image
- one binary column per pathology label

Example:
python scripts/evaluate_folder.py --image-dir data/images --labels data/labels.csv --out outputs/eval.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from xraymind.bootstrap import bootstrap_ci
from xraymind.config import DEFAULT_MODEL_NAME
from xraymind.inference import predict_image
from xraymind.metrics import compute_binary_metrics, tune_threshold
from xraymind.plots import save_reliability_diagram


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate XRayMind on a labeled image folder")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", default="outputs/eval.csv")
    parser.add_argument("--predictions-out", default="outputs/predictions.csv")
    parser.add_argument("--plot-dir", default="outputs/reliability_plots")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tune-thresholds", action="store_true")
    parser.add_argument("--threshold-objective", choices=["f1", "youden"], default="f1")
    parser.add_argument("--bootstrap", type=int, default=0, help="Number of bootstrap samples for AUROC/AUPRC CIs")
    parser.add_argument("--n-bins", type=int, default=10, help="Bins for ECE and reliability plots")
    parser.add_argument("--save-plots", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_dir = Path(args.image_dir)
    labels_df = pd.read_csv(args.labels)
    if args.limit:
        labels_df = labels_df.head(args.limit)

    records = []
    for _, row in labels_df.iterrows():
        image_path = image_dir / str(row[args.image_column])
        prediction = predict_image(image_path, model_name=args.model, top_k=100)
        probs = {p["label"]: p["probability"] for p in prediction["predictions"]}
        records.append({args.image_column: row[args.image_column], **probs})

    pred_df = pd.DataFrame(records)
    predictions_out = Path(args.predictions_out)
    predictions_out.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(predictions_out, index=False)

    metric_rows = []
    label_columns = [c for c in labels_df.columns if c != args.image_column]
    for label in label_columns:
        if label not in pred_df.columns:
            continue
        y_true = labels_df[label].astype(int).to_numpy()
        y_score = pred_df[label].astype(float).to_numpy()
        if len(set(y_true.tolist())) < 2:
            continue

        threshold = (
            tune_threshold(y_true, y_score, objective=args.threshold_objective)
            if args.tune_thresholds
            else args.threshold
        )
        row = compute_binary_metrics(
            label=label,
            y_true=y_true,
            y_score=y_score,
            threshold=threshold,
            n_bins=args.n_bins,
        ).as_dict()

        if args.bootstrap > 0:
            auroc_low, auroc_high = bootstrap_ci(
                y_true, y_score, roc_auc_score, n_bootstrap=args.bootstrap
            )
            auprc_low, auprc_high = bootstrap_ci(
                y_true, y_score, average_precision_score, n_bootstrap=args.bootstrap
            )
            row.update(
                {
                    "auroc_ci_low": auroc_low,
                    "auroc_ci_high": auroc_high,
                    "auprc_ci_low": auprc_low,
                    "auprc_ci_high": auprc_high,
                }
            )

        if args.save_plots:
            safe_label = label.replace("/", "_").replace(" ", "_")
            plot_path = Path(args.plot_dir) / f"{safe_label}_reliability.png"
            save_reliability_diagram(
                y_true,
                y_score,
                plot_path,
                n_bins=args.n_bins,
                title=f"{label} reliability",
            )
            row["reliability_plot"] = str(plot_path)

        metric_rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(out, index=False)
    print(f"Saved prediction table to {predictions_out}")
    print(f"Saved evaluation metrics to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

from xraymind.config import DEFAULT_MODEL_NAME
from xraymind.inference import predict_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate XRayMind on a labeled image folder")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", default="outputs/eval.csv")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--limit", type=int, default=None)
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
    metric_rows = []
    label_columns = [c for c in labels_df.columns if c != args.image_column]
    for label in label_columns:
        if label not in pred_df.columns:
            continue
        y_true = labels_df[label].astype(int).to_numpy()
        y_score = pred_df[label].astype(float).to_numpy()
        if len(set(y_true.tolist())) < 2:
            continue
        metric_rows.append(
            {
                "label": label,
                "auroc": roc_auc_score(y_true, y_score),
                "auprc": average_precision_score(y_true, y_score),
                "n": len(y_true),
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(out, index=False)
    print(f"Saved evaluation metrics to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

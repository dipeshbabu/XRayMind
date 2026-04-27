"""Tune per-label thresholds from a prediction table and labels CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from xraymind.metrics import tune_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune XRayMind thresholds per label")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", default="outputs/thresholds.csv")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--objective", choices=["f1", "youden"], default="f1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels_df = pd.read_csv(args.labels)
    pred_df = pd.read_csv(args.predictions)

    rows = []
    for label in [c for c in labels_df.columns if c != args.image_column]:
        if label not in pred_df.columns:
            continue
        y_true = labels_df[label].astype(int).to_numpy()
        y_score = pred_df[label].astype(float).to_numpy()
        if len(set(y_true.tolist())) < 2:
            continue
        rows.append(
            {
                "label": label,
                "threshold": tune_threshold(y_true, y_score, objective=args.objective),
                "objective": args.objective,
                "n": len(y_true),
                "prevalence": float(y_true.mean()),
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved tuned thresholds to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

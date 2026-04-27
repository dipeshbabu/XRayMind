"""Generate a lightweight model card from XRayMind evaluation metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from xraymind.config import DISCLAIMER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a markdown model card")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default="outputs/MODEL_CARD.md")
    parser.add_argument("--dataset", default="unspecified evaluation set")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.metrics)
    summary = df[["label", "n", "prevalence", "auroc", "auprc", "brier", "ece", "threshold"]].to_markdown(index=False)

    content = f"""# XRayMind Model Card: {args.model}

## Intended use

This model card documents a research evaluation of `{args.model}` inside XRayMind on `{args.dataset}`.

**Safety notice:** {DISCLAIMER}

## Evaluation summary

{summary}

## Metrics

- AUROC and AUPRC evaluate ranking quality.
- Brier score and ECE evaluate probabilistic reliability/calibration.
- Thresholds are validation-set operating points and should not be reused clinically without external validation.

## Limitations

- Image-level labels may be noisy and do not confirm spatial disease localization.
- Performance may shift across hospitals, scanners, patient populations, preprocessing pipelines, and label definitions.
- Heatmaps are model explanations, not radiologist-confirmed findings.
- This repository does not implement regulatory, PACS, DICOM de-identification, or clinical safety workflows.

## Responsible use

Use this card for research comparison, demos, and reproducibility notes only. Do not use the model for clinical diagnosis or treatment decisions.
"""

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    print(f"Saved model card to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

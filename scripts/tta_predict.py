"""Run test-time augmentation prediction for one image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from xraymind.config import DEFAULT_MODEL_NAME, DEFAULT_TOP_K
from xraymind.tta import predict_with_tta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run XRayMind TTA uncertainty prediction")
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", default="outputs/tta_prediction.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = predict_with_tta(
        image=args.image,
        model_name=args.model,
        top_k=args.top_k,
        threshold=args.threshold,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved TTA prediction to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

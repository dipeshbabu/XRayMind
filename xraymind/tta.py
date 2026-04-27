"""Test-time augmentation uncertainty utilities for XRayMind."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image, ImageOps

from .inference import predict_image
from .preprocessing import ImageLike, load_image


def _variants(image: ImageLike) -> List[Image.Image]:
    arr = load_image(image)
    base = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    return [
        base,
        ImageOps.mirror(base),
        base.rotate(-3, resample=Image.BILINEAR, fillcolor=0),
        base.rotate(3, resample=Image.BILINEAR, fillcolor=0),
    ]


def predict_with_tta(
    image: ImageLike,
    model_name: str,
    top_k: int = 5,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Run simple TTA and return mean/std prediction scores.

    This is intended as a rough uncertainty signal for research demos, not a
    calibrated clinical uncertainty estimator.
    """

    predictions = []
    labels = None
    for variant in _variants(image):
        result = predict_image(variant, model_name=model_name, top_k=100, threshold=threshold)
        rows = result["predictions"]
        if labels is None:
            labels = [row["label"] for row in rows]
        predictions.append([row["probability"] for row in rows])

    scores = np.asarray(predictions, dtype=float)
    labels = labels or []
    mean_scores = scores.mean(axis=0)
    std_scores = scores.std(axis=0)
    rows = [
        {
            "label": label,
            "probability_mean": round(float(mean), 6),
            "probability_std": round(float(std), 6),
            "positive": bool(mean >= threshold),
        }
        for label, mean, std in zip(labels, mean_scores, std_scores)
    ]
    rows.sort(key=lambda item: item["probability_mean"], reverse=True)
    return {
        "schema_version": "xraymind.tta_prediction.v1",
        "model": model_name,
        "threshold": threshold,
        "num_augmentations": int(scores.shape[0]),
        "top_findings": rows[:top_k],
        "predictions": rows,
        "uncertainty_summary": {
            "mean_top_std": round(float(np.mean([r["probability_std"] for r in rows[:top_k]])), 6) if rows else 0.0,
            "max_top_std": round(float(np.max([r["probability_std"] for r in rows[:top_k]])), 6) if rows else 0.0,
        },
    }

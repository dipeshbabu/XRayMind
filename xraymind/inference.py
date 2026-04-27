"""Single-image inference helpers for XRayMind."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .config import DISCLAIMER, PredictionConfig
from .model_loader import load_model
from .preprocessing import ImageLike, prepare_xray_tensor


def _as_probabilities(raw: np.ndarray) -> np.ndarray:
    """Return bounded prediction scores.

    TorchXRayVision DenseNet models usually expose probabilities. This helper is
    intentionally conservative and clips values into a valid probability range.
    """

    return np.clip(raw.astype(float), 0.0, 1.0)


def predict_image(
    image: ImageLike,
    model_name: str = PredictionConfig.model_name,
    top_k: int = PredictionConfig.top_k,
    threshold: float = PredictionConfig.threshold,
) -> Dict[str, Any]:
    """Run multi-label chest X-ray prediction and return a structured result."""

    model = load_model(model_name)
    tensor = prepare_xray_tensor(image)

    with torch.no_grad():
        raw = model(tensor)[0].detach().cpu().numpy()

    probabilities = _as_probabilities(raw)
    labels: List[str] = list(model.pathologies)
    rows = []
    for label, score in zip(labels, probabilities):
        if label == "":
            continue
        rows.append(
            {
                "label": label,
                "probability": round(float(score), 6),
                "positive": bool(score >= threshold),
            }
        )

    rows.sort(key=lambda item: item["probability"], reverse=True)
    top_findings = rows[:top_k]
    max_probability = float(top_findings[0]["probability"]) if top_findings else 0.0

    return {
        "schema_version": "xraymind.prediction.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "threshold": threshold,
        "top_k": top_k,
        "top_findings": top_findings,
        "predictions": rows,
        "uncertainty": {
            "max_probability": round(max_probability, 6),
            "low_confidence": bool(max_probability < 0.5),
        },
        "disclaimer": DISCLAIMER,
    }


def save_prediction(result: Dict[str, Any], output_path: str | Path) -> Path:
    """Write a prediction dictionary to JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return path


def predict_to_json(
    image: ImageLike,
    output_path: str | Path,
    model_name: str = PredictionConfig.model_name,
    top_k: int = PredictionConfig.top_k,
    threshold: float = PredictionConfig.threshold,
) -> Path:
    """Run prediction and save the result as JSON."""

    result = predict_image(
        image=image, model_name=model_name, top_k=top_k, threshold=threshold
    )
    return save_prediction(result, output_path)

"""Explainability methods for XRayMind."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from captum.attr import IntegratedGradients, InputXGradient, Saliency

from .model_loader import load_model
from .preprocessing import ImageLike, prepare_xray_tensor


_ATTRIBUTORS = {
    "saliency": Saliency,
    "input_x_gradient": InputXGradient,
    "integrated_gradients": IntegratedGradients,
}


def compute_attribution(
    image: ImageLike,
    label: str,
    model_name: str,
    method: str = "integrated_gradients",
) -> np.ndarray:
    """Compute a normalized attribution heatmap for a target pathology."""

    if method not in _ATTRIBUTORS:
        raise ValueError(f"Unknown method '{method}'. Choose from {sorted(_ATTRIBUTORS)}")

    model = load_model(model_name)
    if label not in model.pathologies:
        raise ValueError(f"Label '{label}' is not available for model '{model_name}'")

    tensor = prepare_xray_tensor(image)
    target_idx = model.pathologies.index(label)
    attr = _ATTRIBUTORS[method](model).attribute(tensor, target=target_idx)
    heatmap = np.abs(attr[0, 0].detach().cpu().numpy())
    heatmap = heatmap - heatmap.min()
    denom = heatmap.max() + 1e-8
    return heatmap / denom


def save_heatmap(
    heatmap: np.ndarray,
    output_path: str | Path,
    cmap: str = "hot",
) -> Path:
    """Save an attribution heatmap as a PNG."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(5, 5))
    plt.imshow(heatmap, cmap=cmap)
    plt.axis("off")
    plt.tight_layout(pad=0)
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return path


def explain_to_file(
    image: ImageLike,
    label: str,
    output_path: str | Path,
    model_name: str,
    method: str = "integrated_gradients",
) -> Path:
    """Compute and save an explanation heatmap for one image and label."""

    heatmap = compute_attribution(
        image=image, label=label, model_name=model_name, method=method
    )
    return save_heatmap(heatmap, output_path)

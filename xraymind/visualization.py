"""Visualization helpers for XRayMind reports."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from matplotlib.figure import Figure
import numpy as np
import torchxrayvision as xrv
from PIL import Image

from .preprocessing import ImageLike, load_image


def _to_grayscale_uint8(image: ImageLike, size: Tuple[int, int] | None = None, *, center_crop: bool = False) -> np.ndarray:
    arr = load_image(image)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    if center_crop:
        arr = xrv.datasets.XRayCenterCrop()(arr[None, ...])[0]
    arr = arr.astype(float)
    arr = arr - arr.min()
    arr = arr / (arr.max() + 1e-8)
    arr = (255 * arr).astype(np.uint8)
    pil = Image.fromarray(arr)
    if size is not None:
        pil = pil.resize(size)
    return np.asarray(pil)


def save_original_preview(image: ImageLike, output_path: str | Path, size: int = 512) -> Path:
    """Save a standardized grayscale preview of the input image."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = _to_grayscale_uint8(image, size=(size, size))
    Image.fromarray(arr).save(path)
    return path


def save_heatmap_overlay(
    image: ImageLike,
    heatmap: np.ndarray,
    output_path: str | Path,
    alpha: float = 0.35,
    size: int = 512,
) -> Path:
    """Save an original-image plus heatmap overlay.

    The overlay is intended for model-debugging and research inspection. It does
    not imply clinically validated localization.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    base = _to_grayscale_uint8(image, size=(size, size), center_crop=True)
    heat = heatmap.astype(float)
    heat = heat - heat.min()
    heat = heat / (heat.max() + 1e-8)
    heat_img = Image.fromarray((255 * heat).astype(np.uint8)).resize((size, size))
    heat = np.asarray(heat_img) / 255.0

    fig = Figure(figsize=(6, 6))
    ax = fig.subplots()
    ax.imshow(base, cmap="gray")
    ax.imshow(heat, cmap="hot", alpha=alpha)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(path, bbox_inches="tight", pad_inches=0, dpi=160)
    return path


def save_side_by_side(
    original_path: str | Path,
    overlay_path: str | Path,
    output_path: str | Path,
    title_left: str = "Original image",
    title_right: str = "Model explanation overlay (center crop)",
) -> Path:
    """Save a two-panel image for report packets."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    original = Image.open(original_path).convert("RGB")
    overlay = Image.open(overlay_path).convert("RGB")

    fig = Figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(original)
    ax1.set_title(title_left)
    ax1.axis("off")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(overlay)
    ax2.set_title(title_right)
    ax2.axis("off")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=160)
    return path

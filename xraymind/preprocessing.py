"""Image loading and preprocessing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
import torchvision
import torchxrayvision as xrv
from PIL import Image

ImageLike = Union[str, Path, np.ndarray, Image.Image]


def load_image(image: ImageLike) -> np.ndarray:
    """Load an image as a numpy array.

    The function accepts a file path, PIL image, or numpy array. Color images are
    allowed and are later converted to a single X-ray channel.
    """

    if isinstance(image, np.ndarray):
        return image
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))

    path = Path(image)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return np.asarray(Image.open(path).convert("RGB"))


def prepare_xray_tensor(image: ImageLike, image_size: int = 224) -> torch.Tensor:
    """Normalize and resize a chest X-ray image for TorchXRayVision models."""

    img = load_image(image)
    img = xrv.datasets.normalize(img, 255)

    if img.ndim > 2:
        img = img.mean(2)[None, ...]
    else:
        img = img[None, ...]

    transform = torchvision.transforms.Compose(
        [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(image_size)]
    )
    img = transform(img)
    tensor = torch.from_numpy(img).float()
    return tensor.unsqueeze(0)

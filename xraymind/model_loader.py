"""Model loading utilities."""

from __future__ import annotations

from functools import lru_cache

import torch
import torchxrayvision as xrv

from .config import DEFAULT_MODEL_NAME


@lru_cache(maxsize=4)
def load_model(model_name: str = DEFAULT_MODEL_NAME) -> torch.nn.Module:
    """Load a TorchXRayVision DenseNet model in eval mode."""

    model = xrv.models.DenseNet(weights=model_name)
    model.eval()
    return model

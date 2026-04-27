"""Shared configuration for XRayMind."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


DISCLAIMER = (
    "Research prototype only. Not for clinical diagnosis or treatment decisions. "
    "Predictions must be reviewed by qualified clinical professionals."
)

DEFAULT_MODEL_NAME = "densenet121-res224-all"
DEFAULT_TOP_K = 5

MODEL_CHOICES: Sequence[str] = (
    "densenet121-res224-all",
    "densenet121-res224-rsna",
    "densenet121-res224-nih",
    "densenet121-res224-pc",
    "densenet121-res224-chex",
    "densenet121-res224-mimic_nb",
    "densenet121-res224-mimic_ch",
)


@dataclass(frozen=True)
class PredictionConfig:
    """Runtime options for single-image inference."""

    model_name: str = DEFAULT_MODEL_NAME
    top_k: int = DEFAULT_TOP_K
    threshold: float = 0.5

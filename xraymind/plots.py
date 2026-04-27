"""Plotting utilities for XRayMind reliability evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def save_reliability_diagram(
    y_true: Iterable[int],
    y_score: Iterable[float],
    output_path: str | Path,
    n_bins: int = 10,
    title: str = "Reliability diagram",
) -> Path:
    """Save a binary reliability diagram."""

    y_true_arr = np.asarray(list(y_true), dtype=float)
    y_score_arr = np.asarray(list(y_score), dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = []
    accuracies = []

    for left, right in zip(bins[:-1], bins[1:]):
        mask = (y_score_arr >= left) & (y_score_arr < right)
        if right == 1.0:
            mask = (y_score_arr >= left) & (y_score_arr <= right)
        if not np.any(mask):
            continue
        centers.append(float(np.mean(y_score_arr[mask])))
        accuracies.append(float(np.mean(y_true_arr[mask])))

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    if centers:
        plt.plot(centers, accuracies, marker="o", label="Model")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path

"""Bootstrap confidence intervals for XRayMind evaluation."""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Tuple

import numpy as np


def bootstrap_ci(
    y_true: Iterable[int],
    y_score: Iterable[float],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[Optional[float], Optional[float]]:
    """Compute a percentile bootstrap confidence interval.

    Samples with only one class are skipped because AUROC/AUPRC may be undefined.
    """

    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_score_arr = np.asarray(list(y_score), dtype=float)
    n = len(y_true_arr)
    if n == 0:
        return None, None

    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true_arr[idx]
        ys = y_score_arr[idx]
        if len(set(yt.tolist())) < 2:
            continue
        try:
            values.append(float(metric_fn(yt, ys)))
        except ValueError:
            continue

    if not values:
        return None, None

    alpha = 1.0 - confidence
    low = float(np.percentile(values, 100 * alpha / 2))
    high = float(np.percentile(values, 100 * (1 - alpha / 2)))
    return low, high

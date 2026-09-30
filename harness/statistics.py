"""Task-level percentile bootstrap intervals for mean benchmark scores.

Resample whole tasks, never individual retrieved documents or judge calls.
These intervals measure task-sampling uncertainty, not variation across repeated
model executions. A constant sample has a degenerate percentile interval.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0


@dataclass(frozen=True)
class MeanInterval:
    estimate: float | None
    low: float | None
    high: float | None
    n: int


def bootstrap_mean(
    scores: Sequence[float],
    *,
    confidence_level: float = CONFIDENCE_LEVEL,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> MeanInterval:
    """Return the mean and a reproducible two-sided percentile bootstrap CI.

    Each resample draws ``len(scores)`` observations with replacement. The
    interval is the central ``confidence_level`` fraction of resampled means.
    Fewer than two observations yields null bounds; nonfinite scores are errors.
    Batches bound memory independently of the number of resamples.
    """
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    if n_resamples < 2:
        raise ValueError("n_resamples must be at least 2")
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("scores must be a one-dimensional sequence of finite numbers")
    n = len(values)
    estimate = float(values.mean()) if n else None
    if n < 2:
        return MeanInterval(estimate, None, None, n)
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    for start in range(0, n_resamples, 256):
        size = min(256, n_resamples - start)
        indices = rng.integers(n, size=(size, n))
        means[start : start + size] = values[indices].mean(axis=1)
    tail = (1 - confidence_level) / 2
    low, high = np.quantile(means, [tail, 1 - tail])
    return MeanInterval(estimate, float(low), float(high), n)

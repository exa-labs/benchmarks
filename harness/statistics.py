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


def bootstrap_ratio(numerators: Sequence[float], denominators: Sequence[float]) -> MeanInterval:
    """Ratio of sums, resampling paired task counts (not individual entities).

    Zero-denominator samples score zero, matching empty-query precision. Use the
    same seed, resample count and confidence level as the default mean intervals.
    """
    top, bottom = np.asarray(numerators, dtype=float), np.asarray(denominators, dtype=float)
    if top.shape != bottom.shape or top.ndim != 1:
        raise ValueError("ratio inputs must be aligned one-dimensional counts")
    if not np.isfinite(top).all() or not np.isfinite(bottom).all() or (bottom < 0).any():
        raise ValueError("ratio counts must be finite with nonnegative denominators")
    n = len(top)
    estimate = float(top.sum() / bottom.sum()) if bottom.sum() else 0.0
    if n < 2:
        return MeanInterval(estimate if n else None, None, None, n)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty(BOOTSTRAP_RESAMPLES)
    for start in range(0, BOOTSTRAP_RESAMPLES, 256):
        size = min(256, BOOTSTRAP_RESAMPLES - start)
        indices = rng.integers(n, size=(size, n))
        den = bottom[indices].sum(axis=1)
        samples[start : start + size] = np.divide(
            top[indices].sum(axis=1),
            den,
            out=np.zeros(size),
            where=den != 0,
        )
    tail = (1 - CONFIDENCE_LEVEL) / 2
    low, high = np.quantile(samples, [tail, 1 - tail])
    return MeanInterval(estimate, float(low), float(high), n)

"""Statistical checks against known distributions and bootstrap invariants."""

import pytest

from harness.statistics import MeanInterval, bootstrap_mean, bootstrap_ratio


def test_binary_mean_matches_exact_binomial_percentiles():
    # Resampling 100 balanced binary scores gives Binomial(100, .5) / 100,
    # whose exact 2.5th and 97.5th percentiles are .40 and .60.
    interval = bootstrap_mean([0.0] * 50 + [1.0] * 50)
    assert interval.estimate == 0.5
    assert interval.low == pytest.approx(0.40, abs=0.01)
    assert interval.high == pytest.approx(0.60, abs=0.01)
    assert interval.n == 100


def test_bootstrap_is_reproducible_and_preserves_affine_transformations():
    scores = [0.0, 0.1, 0.4, 0.6, 0.9] * 20
    original = bootstrap_mean(scores, seed=42)
    assert original == bootstrap_mean(scores, seed=42)
    transformed = bootstrap_mean([3 * value + 2 for value in scores], seed=42)
    assert transformed.estimate == pytest.approx(3 * original.estimate + 2)
    assert transformed.low == pytest.approx(3 * original.low + 2)
    assert transformed.high == pytest.approx(3 * original.high + 2)


@pytest.mark.parametrize("scores,estimate", [([], None), ([0.7], 0.7)])
def test_insufficient_data_has_no_interval(scores, estimate):
    assert bootstrap_mean(scores) == MeanInterval(estimate, None, None, len(scores))


def test_constant_scores_have_a_degenerate_interval():
    assert bootstrap_mean([1.0] * 20) == MeanInterval(1.0, 1.0, 1.0, 20)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_scores_are_rejected(bad):
    with pytest.raises(ValueError, match="finite"):
        bootstrap_mean([0.0, bad])


@pytest.mark.parametrize("level", [0, 1, -0.1, float("nan")])
def test_invalid_confidence_levels_are_rejected(level):
    with pytest.raises(ValueError, match="confidence_level"):
        bootstrap_mean([0.0, 1.0], confidence_level=level)


def test_at_least_two_resamples_are_required():
    with pytest.raises(ValueError, match="n_resamples"):
        bootstrap_mean([0.0, 1.0], n_resamples=1)


def test_ratio_bootstrap_weights_counts_and_resamples_pairs():
    interval = bootstrap_ratio([1, 0], [1, 9])
    assert interval.estimate == 0.1  # entity weighted, not the 0.5 mean task precision
    assert interval.low == 0.0 and interval.high == 1.0
    assert interval == bootstrap_ratio([2, 0], [2, 18])
    assert bootstrap_ratio([1, 9], [1, 9]) == MeanInterval(1.0, 1.0, 1.0, 2)


def test_ratio_bootstrap_handles_empty_and_zero_entity_samples():
    assert bootstrap_ratio([], []) == MeanInterval(None, None, None, 0)
    assert bootstrap_ratio([0], [0]) == MeanInterval(0.0, None, None, 1)
    assert bootstrap_ratio([0, 0], [0, 0]) == MeanInterval(0.0, 0.0, 0.0, 2)

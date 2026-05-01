"""Tests for helper functions related to sampling random segment sets matching target segments."""

import numpy as np

from experiments.faithfulness.src.helpers import (
    quantile_bins,
    sample_random_set_matching_targets,
)


def test_quantile_bins_handles_constant_values() -> None:
    """Test that quantile_bins can handle constant values without error
    and returns a single bin."""
    bins = quantile_bins([2.0, 2.0, 2.0, 2.0], n_bins=4)
    assert bins.tolist() == [0, 0, 0, 0]


def test_sample_random_set_matching_targets_excludes_forbidden_indices() -> None:
    """Test that sample_random_set_matching_targets correctly excludes specified indices
    and returns the expected number of random picks."""
    rng = np.random.default_rng(123)
    duration_bins = np.asarray([0, 0, 1, 1, 2, 2], dtype=int)
    position_bins = np.asarray([0, 1, 0, 1, 0, 1], dtype=int)

    picked = sample_random_set_matching_targets(
        target_indices=[0, 2],
        n=6,
        duration_bins=duration_bins,
        position_bins=position_bins,
        rng=rng,
        baseline_type="stratified_random",
        global_exclude={0, 2},
    )

    assert len(picked) == 2
    assert len(set(picked)) == 2
    assert 0 not in picked
    assert 2 not in picked


def test_sample_random_set_matching_targets_uniform_mode() -> None:
    """Test that sample_random_set_matching_targets in uniform_random mode returns random picks
    that are not in the target indices and respects the global exclusion set."""
    rng = np.random.default_rng(42)
    duration_bins = np.asarray([0, 1, 2, 3], dtype=int)
    position_bins = np.asarray([0, 1, 0, 1], dtype=int)

    picked = sample_random_set_matching_targets(
        target_indices=[1],
        n=4,
        duration_bins=duration_bins,
        position_bins=position_bins,
        rng=rng,
        baseline_type="uniform_random",
        global_exclude={1},
    )

    assert len(picked) == 1
    assert picked[0] in {0, 2, 3}

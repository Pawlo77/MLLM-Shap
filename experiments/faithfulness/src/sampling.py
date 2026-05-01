"""Random sampling utilities for faithfulness evaluation baselines."""

from collections.abc import Sequence

import numpy as np


def quantile_bins(values: Sequence[float], n_bins: int) -> np.ndarray:
    """Discretize values into quantile bins with safe fallbacks.

    If values are constant or insufficiently diverse, returns a single zero bin.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.asarray([], dtype=int)
    if n_bins <= 1:
        return np.zeros(arr.shape[0], dtype=int)
    cuts = np.unique(np.quantile(arr, np.linspace(0.0, 1.0, n_bins + 1)))
    if cuts.size <= 2:
        return np.zeros(arr.shape[0], dtype=int)
    return np.digitize(arr, cuts[1:-1], right=True).astype(int)


def sample_uniform_index(
    n: int,
    exclude: set[int],
    rng: np.random.Generator,
) -> int:
    """Sample one index uniformly from ``range(n)`` excluding forbidden indices."""
    candidates = [idx for idx in range(n) if idx not in exclude]
    if not candidates:
        raise ValueError("No candidates left for uniform random sampling.")
    return int(rng.choice(candidates))


def sample_stratified_index(
    target_idx: int,
    duration_bins: np.ndarray,
    position_bins: np.ndarray,
    exclude: set[int],
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Sample one index with duration/position matching to a target segment.

    Returns ``(sampled_idx, match_level)`` where ``match_level`` encodes how
    closely the sample matched the target bins:

    * 0 – strict (duration **and** position bin match)
    * 1 – duration-only match
    * 2 – position-only match
    * 3 – uniform fallback (no bin matched; occurs when few segments cause bin
          collapse — a caller-visible signal that stratified quality degraded)

    Sampling prefers strict matches and progressively relaxes when not
    enough candidates are available.
    """
    n = len(duration_bins)
    candidates = np.asarray([idx for idx in range(n) if idx not in exclude], dtype=int)
    if candidates.size == 0:
        raise ValueError("No candidates left for stratified random sampling.")

    strict = candidates[
        (duration_bins[candidates] == duration_bins[target_idx])
        & (position_bins[candidates] == position_bins[target_idx])
    ]
    if strict.size:
        return int(rng.choice(strict)), 0

    dur_only = candidates[duration_bins[candidates] == duration_bins[target_idx]]
    if dur_only.size:
        return int(rng.choice(dur_only)), 1

    pos_only = candidates[position_bins[candidates] == position_bins[target_idx]]
    if pos_only.size:
        return int(rng.choice(pos_only)), 2

    return int(rng.choice(candidates)), 3


def sample_random_set_matching_targets(
    target_indices: Sequence[int],
    n: int,
    duration_bins: np.ndarray,
    position_bins: np.ndarray,
    rng: np.random.Generator,
    baseline_type: str,
    global_exclude: set[int] | None = None,
) -> tuple[list[int], float | None]:
    """Sample a random index set aligned to a target set under a baseline policy.

    Each target index gets one random counterpart and sampled indices are unique.

    Returns ``(selected_indices, strict_match_rate)`` where ``strict_match_rate``
    is the fraction of stratified draws that achieved strict (duration+position)
    bin matching (level 0). A value below 1.0 indicates that bin collapse forced
    relaxed or uniform fallback — a signal that §5.1 bias mitigation degraded.
    For ``"uniform_random"`` baseline, ``strict_match_rate`` is ``None``.
    """
    selected: list[int] = []
    used = set(global_exclude or set())
    strict_count = 0
    n_stratified = 0
    for target_idx in target_indices:
        if baseline_type == "uniform_random":
            rand_idx = sample_uniform_index(n=n, exclude=used, rng=rng)
        elif baseline_type == "stratified_random":
            rand_idx, level = sample_stratified_index(
                target_idx=int(target_idx),
                duration_bins=duration_bins,
                position_bins=position_bins,
                exclude=used,
                rng=rng,
            )
            n_stratified += 1
            if level == 0:
                strict_count += 1
        else:
            raise ValueError(f"Unsupported baseline_type={baseline_type!r}")
        selected.append(int(rand_idx))
        used.add(int(rand_idx))
    strict_match_rate: float | None = (
        strict_count / n_stratified if n_stratified > 0 else None
    )
    return selected, strict_match_rate

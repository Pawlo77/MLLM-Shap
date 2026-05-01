"""General helper utilities for faithfulness evaluation.

This module re-exports symbols from the focused submodules for backward
compatibility. New code should import from the specific submodule directly.
"""

import math

import numpy as np
from scipy import stats

from .audio import (
    aggregate_sv_to_segments,
    extract_audio_sv,
    remove_interval,
    segment_interval,
)
from .io import (
    experiment_set_from_spec,
    load_selected_rows,
    load_spec,
    parse_sample_id,
    sample_paths,
)
from .sampling import (
    quantile_bins,
    sample_random_set_matching_targets,
    sample_stratified_index,
    sample_uniform_index,
)
from .similarity import (
    embedding_similarities,
    generate_response,
    sequence_match_similarities,
    tfidf_similarities,
)

EPS: float = 1e-9
"""Small constant used to avoid divide-by-zero and numerical instabilities."""


def response_drop(full_similarity: float, perturbed_similarity: float) -> float:
    """Return similarity drop induced by a perturbation.

    Positive values indicate that the perturbation reduced similarity.
    """
    return float(full_similarity - perturbed_similarity)


def as_list(value) -> list:
    """Normalize a scalar/array/list value into a Python list."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def estimate_required_paired_n(
    target_effect_size_dz: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int | None:
    """Approximate required n for a paired t-test using normal approximation.

    This is a planning approximation only and should be treated as conservative.
    """
    if target_effect_size_dz <= 0.0:
        return None
    if not (0.0 < alpha < 1.0 and 0.0 < power < 1.0):
        return None

    z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
    z_beta = stats.norm.ppf(power)
    required = ((z_alpha + z_beta) / target_effect_size_dz) ** 2
    return int(math.ceil(required))


def rank_abs_sv(segment_sv_values: list[float]) -> dict[str, object]:
    """Compute absolute-SV ranking and concentration diagnostics for segments."""
    values = np.asarray(segment_sv_values, dtype=float)
    abs_values = np.abs(values)
    order = np.argsort(-abs_values, kind="mergesort")
    ranks = np.empty(len(abs_values), dtype=int)
    ranks[order] = np.arange(1, len(abs_values) + 1)

    total_abs = float(abs_values.sum())
    shares = abs_values / (total_abs + EPS)
    top_abs = float(abs_values[order[0]]) if len(order) else 0.0
    second_abs = float(abs_values[order[1]]) if len(order) > 1 else None
    top1_top2_gap = top_abs - second_abs if second_abs is not None else None
    top1_top2_ratio = top_abs / (second_abs + EPS) if second_abs is not None else None
    top1_share = float(shares[order[0]]) if len(order) else 0.0

    positive_shares = shares[shares > 0]
    entropy_norm = None
    if len(shares) > 1 and len(positive_shares):
        entropy = -float(np.sum(positive_shares * np.log(positive_shares)))
        entropy_norm = entropy / float(np.log(len(shares)))

    sorted_abs = np.sort(abs_values)
    if len(sorted_abs) == 0 or total_abs <= EPS:
        gini = 0.0
    else:
        index = np.arange(1, len(sorted_abs) + 1)
        gini = float(
            (2 * np.sum(index * sorted_abs)) / (len(sorted_abs) * total_abs)
            - (len(sorted_abs) + 1) / len(sorted_abs)
        )

    return {
        "order": order,
        "ranks": ranks,
        "abs_values": abs_values,
        "shares": shares,
        "top_abs": top_abs,
        "top1_top2_gap": float(top1_top2_gap) if top1_top2_gap is not None else None,
        "top1_top2_ratio": float(top1_top2_ratio)
        if top1_top2_ratio is not None
        else None,
        "top1_share": top1_share,
        "abs_sv_entropy_norm": float(entropy_norm)
        if entropy_norm is not None
        else None,
        "abs_sv_gini": gini,
    }


__all__ = [
    "EPS",
    "aggregate_sv_to_segments",
    "as_list",
    "embedding_similarities",
    "estimate_required_paired_n",
    "experiment_set_from_spec",
    "extract_audio_sv",
    "generate_response",
    "load_selected_rows",
    "load_spec",
    "parse_sample_id",
    "quantile_bins",
    "rank_abs_sv",
    "remove_interval",
    "response_drop",
    "sample_paths",
    "sample_random_set_matching_targets",
    "sample_stratified_index",
    "sample_uniform_index",
    "segment_interval",
    "sequence_match_similarities",
    "tfidf_similarities",
]

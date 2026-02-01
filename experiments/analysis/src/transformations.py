"""Transformations for experiments analysis."""

import numpy as np
import pandas as pd
from scipy.stats import entropy


def interpolate_cumsum(values: list[float], target_len: int) -> list[float]:
    """Interpolate cumulative shap values to a target length."""
    values = np.asarray(values, dtype=float)
    n = len(values)

    # Only one point -> repeat it
    if n == 1:
        return values.repeat(target_len).tolist()

    # Original and target normalized positions
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, target_len)

    # Linear interpolation
    y = np.interp(x_new, x_old, values)

    # Ensure cumulative (monotonic increasing)
    y = np.maximum.accumulate(y)

    # Renormalize so final value matches the original cumsum
    if y[-1] > 0:
        y *= values[-1] / y[-1]

    return y.tolist()


def normalize_lengths(group: pd.DataFrame, col="sv_cumsum") -> pd.DataFrame:
    """Normalize lengths of cumulative shap values within a group."""
    target_len = group[col].apply(len).max()
    group[col] = group[col].apply(lambda v: interpolate_cumsum(v, target_len))
    return group


def calculate_normalized_entropy(sv_list: list) -> float:
    """Calculate Normalized Entropy of Shapley values.

    Uses absolute values of Shapley values to treat negative and positive contributions
    as equally 'important' for the distribution shape.
    """
    sv_list = np.abs(np.array(sv_list, dtype=float))

    # Normalize to probability distribution
    total = np.sum(sv_list)
    if total == 0:
        return 1.0

    probs = sv_list / total

    h_val = entropy(probs)
    n = len(sv_list)
    if n <= 1:
        return 0.0

    max_entropy = np.log(n)
    return h_val / max_entropy

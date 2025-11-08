"""Utility functions for Monte Carlo sampling in SHAP explanations."""

import math


def approximate_budget(error_bound: float, confidence: float) -> int:
    """
    Calculate the approximate number of samples needed to achieve a desired error bound
    with a specified confidence level using Hoeffding's inequality.

    Args:
        error_bound (float): The maximum allowable error.
        confidence (float): The desired confidence level (between 0 and 1).
    Returns:
        int: The calculated number of samples needed.
    """
    return math.ceil(2 * math.log(2 / (1 - confidence)) / (error_bound**2))

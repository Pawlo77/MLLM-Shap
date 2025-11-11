# pylint: disable=too-few-public-methods

"""Normalizers for SHAP values."""

import torch
from torch import Tensor

from .base.normalizers import BaseNormalizer


class IdentityNormalizer(BaseNormalizer):
    """Normalizer that returns SHAP values unchanged."""

    def __call__(self, shap_values: Tensor) -> Tensor:
        return shap_values


class AbsSumNormalizer(BaseNormalizer):
    """Normalizer that scales SHAP values by the sum of their absolute values."""

    def __call__(self, shap_values: Tensor) -> Tensor:
        abs_sum = shap_values.abs().sum()
        if abs_sum == 0:
            return shap_values
        return shap_values / abs_sum


class PowerShiftNormalizer(BaseNormalizer):
    """Normalizer that applies power shift normalization to SHAP values."""

    power: float
    """The power to which SHAP values are raised."""

    def __init__(self, power: float = 1.0):
        """
        Initialize the PowerShiftNormalizer.

        Args:
            power: The power to which SHAP values are raised.
        Raises:
            ValueError: If power is not a positive float.
        """
        if isinstance(power, (int, float)) and power <= 0:
            raise ValueError("power must be a positive float.")
        self.power = power

    def __call__(self, shap_values: Tensor) -> Tensor:
        # Shift SHAP values to be non-negative
        shifted = shap_values - shap_values.min()

        # Apply power transformation
        powered = shifted.pow(self.power)

        # Normalize to sum to 1
        total = powered.sum()
        if total == 0:
            return shap_values
        normalized = powered / total
        return normalized


class MinMaxNormalizer(BaseNormalizer):
    """
    Normalizer that scales SHAP values to the [0, 1] range using min-max normalization
    then normalizes them to sum to 1.
    """

    def __call__(self, shap_values: Tensor) -> Tensor:
        min_val = shap_values.min()
        max_val = shap_values.max()
        if max_val - min_val == 0:  # all values are the same
            return torch.ones_like(shap_values) / len(shap_values)
        normalized = (shap_values - min_val) / (max_val - min_val)
        normalized /= normalized.sum()
        return normalized

"""Base Monte Carlo approximation SHAP explainer implementation."""

from abc import ABC
from functools import lru_cache
import torch
from torch import Tensor

from ..base.approx import BaseShapApproximation


# pylint: disable=too-few-public-methods
class BaseMcShapExplainer(BaseShapApproximation, ABC):
    """Base Monte Carlo SHAP implementation class"""

    @lru_cache(maxsize=1)
    def _get_num_splits(self, n: int) -> int:
        if self.num_samples is not None:
            if self.num_samples == -1:
                # Minimal: only single-feature masks and empty mask
                return n + 1
            if self.num_samples < n:
                raise ValueError("num_samples must be at least equal to the number of features.")
            if self.num_samples > (2**n - 1):
                return int(2**n - 1)  # maximum possible masks excluding all-ones mask
            return self.num_samples

        total_masks = 2**n - 1  # exclude all-ones mask
        return int(total_masks * self.fraction)

    # pylint: disable=unused-argument
    def _calculate_shap_values(self, masks: Tensor, similarities: Tensor, device: torch.device) -> Tensor:
        included_mean = (masks * similarities[:, None]).sum(dim=0) / masks.sum(dim=0)
        excluded_mean = ((~masks) * similarities[:, None]).sum(dim=0) / (~masks).sum(dim=0)
        return included_mean - excluded_mean

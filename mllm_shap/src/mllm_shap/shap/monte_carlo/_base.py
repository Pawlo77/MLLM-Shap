"""Base Monte Carlo approximation SHAP explainer implementation."""

from abc import ABC
from functools import lru_cache
import torch
from torch import Tensor

from ..base.approx import BaseShapApproximation


# pylint: disable=too-few-public-methods
class BaseMcShapExplainer(BaseShapApproximation, ABC):
    """Base Monte Carlo SHAP implementation class"""

    include_minimal_masks: bool = True
    """Whether to include minimal masks (single-feature and empty masks) in the sampling."""

    @lru_cache(maxsize=1)
    def _get_num_splits(self, target_length: int) -> int:
        if self.num_samples is not None:
            if self.num_samples == -1:
                # Minimal: only single-feature masks and empty mask
                return target_length + 1
            if self.num_samples < target_length:
                raise ValueError("num_samples must be at least equal to the number of features.")
            if self.num_samples > (2**target_length - 1):
                return int(2**target_length - 1)  # maximum possible masks excluding all-ones mask
            return self.num_samples

        total_masks = 2**target_length - 1  # exclude all-ones mask
        return int(total_masks * self.fraction)

    def _get_next_split(self, target_length: int, device: torch.device, generated_masks: int) -> Tensor | None:
        r = self._get_next_split_base(target_length=target_length, device=device, generated_masks=generated_masks)
        if r is not None:
            return r

        if generated_masks < self._get_num_splits(target_length):
            return torch.randint(0, 2, (1, target_length), dtype=torch.bool, device=device)
        return None

    # pylint: disable=unused-argument
    def _calculate_shap_values(self, masks: Tensor, similarities: Tensor, device: torch.device) -> Tensor:
        included_mean = (masks * similarities[:, None]).sum(dim=0) / masks.sum(dim=0)
        excluded_mean = ((~masks) * similarities[:, None]).sum(dim=0) / (~masks).sum(dim=0)
        return included_mean - excluded_mean

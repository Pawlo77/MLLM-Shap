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

    __base_masks: Tensor | None

    @lru_cache(maxsize=1)
    def _get_num_splits(self, target_length: int) -> int:
        """
        Determine the number of masks to generate based on num_samples and fraction.

        Args:
            target_length: Length of the masks
        Returns:
            Number of masks to generate.
        """
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
        if self.include_minimal_masks:
            if generated_masks == 0:
                self.__base_masks = BaseMcShapExplainer.__generate_minimal_splits(
                    target_length=target_length,
                    device=device,
                )
            if self.__base_masks is None:
                raise RuntimeError("Base masks are not present.")
            if self._get_num_splits(target_length) < self.__base_masks.shape[0]:
                raise RuntimeError(
                    f"Not enough sampling budget, up to {self._get_num_splits(target_length)} "
                    f"calls allowed with required {self.__base_masks.shape[0]} for minimal masks."
                )

            if generated_masks < self.__base_masks.shape[0]:
                return self.__base_masks[generated_masks, ...].squeeze(0)

        if generated_masks < self._get_num_splits(target_length):
            return torch.randint(0, 2, (1, target_length), dtype=torch.bool, device=device)
        return None

    # pylint: disable=unused-argument
    def _calculate_shap_values(self, masks: Tensor, similarities: Tensor, device: torch.device) -> Tensor:
        included_mean = (masks * similarities[:, None]).sum(dim=0) / masks.sum(dim=0)
        excluded_mean = ((~masks) * similarities[:, None]).sum(dim=0) / (~masks).sum(dim=0)
        return included_mean - excluded_mean

    @staticmethod
    def __generate_minimal_splits(target_length: int, device: torch.device) -> torch.Tensor:
        """
        Generate a minimal set of boolean masks as a batched tensor.
        Shape: (target_length + 1, target_length)
        """
        masks = torch.ones((target_length + 1, target_length), dtype=torch.bool, device=device)
        masks[0, :] = False
        masks[torch.arange(1, target_length + 1), torch.arange(target_length)] = False
        return masks

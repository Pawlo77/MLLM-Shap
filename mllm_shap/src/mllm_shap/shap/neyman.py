"""Complementary Neyman SHAP explainer implementation."""

import torch
from torch import Tensor

from .base.approx import BaseShapApproximation


# pylint: disable=too-few-public-methods
class ComplementaryNeymanShapExplainer(BaseShapApproximation):
    """Base Complementary Neyman SHAP implementation class"""

    # TODO

    def _get_num_splits(self, target_length: int) -> int | None:
        """
        Determine the number of masks to generate based on num_samples and fraction.

        Args:
            n: Length of the masks
        Returns:
            Number of masks to generate.
        """
        if self.num_samples is not None:
            if self.num_samples == -1:
                # Minimal: only single-feature masks, their negations and empty mask
                return 2 * target_length + 1
            if self.num_samples < 2 * target_length:
                raise ValueError("num_samples must be at least equal to the number of features times two.")
            if self.num_samples > (2**target_length - 1):
                return int(2**target_length - 1)  # maximum possible masks excluding all-ones
            if self.num_samples % 2 == 0:
                raise ValueError(
                    "num_samples must be odd to account for "
                    "complementary masks (1 for empty mask, remaining in pairs)."
                )
            return self.num_samples

        total_masks = 2**target_length - 1  # exclude all-ones mask
        return int(total_masks * self.fraction)

    def _get_next_split(self, target_length: int, device: torch.device, generated_masks: int) -> Tensor | None:
        return None

    def _calculate_shap_values(
        self,
        masks: Tensor,
        similarities: Tensor,
        device: torch.device,
    ) -> Tensor:
        return torch.zeros((masks.shape[1],), device=device)

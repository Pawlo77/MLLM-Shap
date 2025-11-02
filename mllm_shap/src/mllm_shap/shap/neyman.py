"""Complementary Neyman SHAP explainer implementation."""

import torch
from torch import Tensor

from .base.approx import BaseShapApproximation


# pylint: disable=too-few-public-methods
class ComplementaryNeymanShapExplainer(BaseShapApproximation):
    """Base Complementary Neyman SHAP implementation class"""

    def _get_num_masks(self, n: int) -> int:
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
                return 2 * n + 1
            if self.num_samples < 2 * n:
                raise ValueError("num_samples must be at least equal to the number of features times two.")
            if self.num_samples > (2**n - 1):
                return 2**n - 1  # maximum possible masks excluding all-ones
            if self.num_samples % 2 == 0:
                raise ValueError(
                    "num_samples must be odd to account for "
                    "complementary masks (1 for empty mask, remaining in pairs)."
                )
            return self.num_samples

        total_masks = 2**n - 1  # exclude all-ones mask
        return int(total_masks * self.fraction)

    # pylint: disable=duplicate-code
    def _generate_masks(self, n: int, device: torch.device, existing_masks: Tensor | None = None) -> Tensor:
        seen: set[tuple[float]] = set()
        mask_list: list[Tensor] = []

        # presence of mask S implies presence of its complementary ~S
        num_masks = self._get_num_masks(n)
        self._mark_existing_masks(existing_masks, seen)
        mask_list = self._generate_minimal_masks(n, device, seen, mask_list)
        # add their negations
        for neg_mask in [~mask for mask in mask_list if mask.sum() != 0]:
            mask_list = type(self)._update_masks(neg_mask, mask_list, seen)

        # generate random unique multi-feature masks
        remaining_masks_needed = num_masks - len(seen)
        while len(mask_list) < remaining_masks_needed:
            break  # TODO

        if not mask_list:
            return torch.empty((0, n), dtype=torch.bool, device=device)
        return torch.stack(mask_list, dim=0)

    def _calculate_shap_values(self, masks: Tensor, similarities: Tensor, device: torch.device) -> Tensor:
        raise NotImplementedError("Complementary SHAP value calculation is not implemented yet.")

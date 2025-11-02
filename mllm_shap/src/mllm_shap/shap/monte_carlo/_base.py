"""Base Monte Carlo approximation SHAP explainer implementation."""

from abc import ABC
import torch
from torch import Tensor

from ..base.approx import BaseShapApproximation


# pylint: disable=too-few-public-methods
class BaseMcShapExplainer(BaseShapApproximation, ABC):
    """Base Monte Carlo SHAP implementation class"""

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
                # Minimal: only single-feature masks and empty mask
                return n + 1
            if self.num_samples < n:
                raise ValueError("num_samples must be at least equal to the number of features.")
            if self.num_samples > (2**n - 1):
                return int(2**n - 1)  # maximum possible masks excluding all-zeros
            return self.num_samples

        total_masks = 2**n - 1  # exclude all-ones mask
        return int(total_masks * self.fraction)

    # pylint: disable=duplicate-code
    def _generate_masks(self, n: int, device: torch.device, existing_masks: Tensor | None = None) -> Tensor:
        seen: set[tuple[float]] = set()
        mask_list: list[Tensor] = []

        num_masks = self._get_num_masks(n)
        self._mark_existing_masks(existing_masks, seen)
        mask_list = self._generate_minimal_masks(n, device, seen, mask_list)

        # generate random unique multi-feature masks
        remaining_masks_needed = num_masks - len(seen)
        while len(mask_list) < remaining_masks_needed:
            mask = torch.randint(0, 2, (n,), dtype=torch.bool, device=device)

            # skip all-zeros and single-feature masks
            if mask.sum() <= 1:
                continue

            mask_list = self._update_masks(mask, mask_list, seen)

        if not mask_list:
            return torch.empty((0, n), dtype=torch.bool, device=device)
        return torch.stack(mask_list, dim=0)

    # pylint: disable=unused-argument
    def _calculate_shap_values(self, masks: Tensor, similarities: Tensor, device: torch.device) -> Tensor:
        included_mean = (masks * similarities[:, None]).sum(dim=0) / masks.sum(dim=0)
        excluded_mean = ((~masks) * similarities[:, None]).sum(dim=0) / (~masks).sum(dim=0)
        return included_mean - excluded_mean

"""Monte Carlo approximation SHAP explainer implementation."""

from typing import Any

import torch
from torch import Tensor

from ._base.explainer import BaseSHAPExplainer


# pylint: disable=too-few-public-methods
class MCSHAPExplainer(BaseSHAPExplainer):
    """
    Monte Carlo SHAP implementation.

    Fields:
        num_samples: Number of random masks to generate. If None, uses fraction. -1 stands for minimal.
        fraction: Fraction of total possible masks to generate if num_samples is None.
    """

    num_samples: int | None
    fraction: float | None

    def __init__(self, *args: Any, num_samples: int | None = None, fraction: float = 0.6, **kwargs: Any) -> None:
        """
        Initialize the MCSHAPExplainer.

        Args:
            num_samples: Number of random masks to generate. If None, uses fraction.
            fraction: Fraction of total possible masks to generate if num_samples is None.
        Raises:
            ValueError: If both num_samples and fraction are None.
        """
        super().__init__(*args, **kwargs)

        if num_samples is None and fraction is None:
            raise ValueError("Either num_samples or fraction must be provided.")
        if fraction is not None and (not isinstance(fraction, float) or not 0 < fraction <= 1):
            raise ValueError("fraction must be a float in the range (0, 1].")
        if num_samples is not None and (not isinstance(num_samples, int) or (num_samples <= 0 and num_samples != -1)):
            raise ValueError("num_samples must be a positive integer.")

        self.num_samples = num_samples
        self.fraction = fraction

    def _generate_masks(self, n: int, device: torch.device, existing_masks: Tensor | None = None) -> Tensor:
        if self.num_samples is not None:
            if self.num_samples == -1:
                # Minimal: only single-feature masks and empty mask
                num_masks = n + 1
            elif self.num_samples < n:
                raise ValueError("num_samples must be at least equal to the number of features.")
            elif self.num_samples > (2**n - 1):
                num_masks = 2**n - 1  # maximum possible masks excluding all-zeros
            else:
                num_masks = self.num_samples
        else:
            total_masks = 2**n - 1  # exclude all-zeros mask
            num_masks = int(total_masks * self.fraction)

        seen: set[tuple[float]] = set()
        mask_list: list[Tensor] = []

        # mark existing masks as seen
        if existing_masks is not None and existing_masks.numel() > 0:
            existing_seen = {tuple(row.tolist()) for row in existing_masks}
            seen |= existing_seen

        # include all-zeros mask
        zero_mask = torch.zeros(n, dtype=torch.bool, device=device)
        mask_list = MCSHAPExplainer.__update_masks_if_not_present(zero_mask, mask_list, seen)

        # include all single-feature masks
        for i in range(n):
            mask = torch.zeros(n, dtype=torch.bool, device=device)
            mask[i] = True
            mask_list = MCSHAPExplainer.__update_masks_if_not_present(mask, mask_list, seen)

        # generate random unique multi-feature masks
        remaining_masks_needed = num_masks - len(seen)
        while len(mask_list) < remaining_masks_needed:
            mask = torch.randint(0, 2, (n,), dtype=torch.bool, device=device)

            # skip all-zeros and single-feature masks
            if mask.sum() <= 1:
                continue

            mask_list = MCSHAPExplainer.__update_masks_if_not_present(mask, mask_list, seen)

        if not mask_list:
            return torch.empty((0, n), dtype=torch.bool, device=device)
        return torch.stack(mask_list, dim=0)

    def _calculate_shap_values(self, masks: Tensor, similarities: Tensor, device: torch.device) -> Tensor:
        included_mean = (masks * similarities[:, None]).sum(dim=0) / masks.sum(dim=0)
        excluded_mean = ((~masks) * similarities[:, None]).sum(dim=0) / (~masks).sum(dim=0)
        return included_mean - excluded_mean

    @staticmethod
    def __update_masks_if_not_present(
        mask: Tensor, masks: list[Tensor], existing_set: set[tuple[float]]
    ) -> list[Tensor]:
        """
        Add mask to masks list if not already present in existing_set.
        Updates existing_set with the new mask key if added.

        Args:
            mask: The mask tensor to potentially add.
            masks: The current list of masks.
            existing_set: Set of existing mask keys for quick lookup.
        Returns:
            Updated list of masks.
        """
        key = tuple(mask.tolist())
        if key not in existing_set:
            existing_set.add(key)
            masks.append(mask)
        return masks

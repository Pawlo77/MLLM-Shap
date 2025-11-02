"""Base class for SHAP explainers using approximation methods."""

from abc import ABC, abstractmethod

from typing import Any

import torch
from torch import Tensor

from .explainer import BaseShapExplainer


# pylint: disable=too-few-public-methods
class BaseShapApproximation(BaseShapExplainer, ABC):
    """
    Base class for SHAP explainers using approximation methods.
    """

    num_samples: int | None
    """
    Number of random masks to generate. If None, uses fraction.
    -1 stands for minimal number of samples (only single-feature masks and empty mask).
    """

    fraction: float | None
    """Fraction of total possible masks to generate if num_samples is None."""

    def __init__(self, *args: Any, num_samples: int | None = None, fraction: float = 0.6, **kwargs: Any) -> None:
        """
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

    @abstractmethod
    def _get_num_masks(self, n: int) -> int:
        """
        Determine the number of masks to generate based on num_samples and fraction.

        Args:
            n: Length of the masks
        Returns:
            Number of masks to generate.
        """

    def _mark_existing_masks(self, existing_masks: Tensor | None, seen: set[tuple[float]]) -> None:
        """
        Get starting masks based on num_samples.

        Args:
            existing_masks: Existing masks to consider.
            seen: Set to populate with existing mask keys.
        """
        if existing_masks is not None and existing_masks.numel() > 0:
            existing_seen = {tuple(row.tolist()) for row in existing_masks}
            seen |= existing_seen

    def _generate_minimal_masks(
        self, n: int, device: torch.device, seen: set[tuple[float]], mask_list: list[Tensor]
    ) -> list[Tensor]:
        """
        Generate minimal set of masks: all single-feature masks and the empty mask.

        Args:
            n: Number of features.
            device: Device to create the masks on.
            seen: Set of existing mask keys for quick lookup.
            mask_list: List to append the generated masks to.
        Returns:
            Updated list of masks.
        """
        # include all-zeros mask
        zero_mask = torch.zeros(n, dtype=torch.bool, device=device)
        mask_list = self._update_masks(zero_mask, mask_list, seen)

        # include all single-feature masks
        for i in range(n):
            mask = torch.zeros(n, dtype=torch.bool, device=device)
            mask[i] = True
            mask_list = self._update_masks(mask, mask_list, seen)

        return mask_list

    @abstractmethod
    def _update_masks(self, *args: Any, **kwargs: Any) -> list[Tensor]:
        """
        Add mask to masks list. Updates existing_set with the new mask key if added.

        Args:
            args: Arguments to pass to the executor.
            kwargs: Keyword arguments to pass to the executor.
        Returns:
            Updated list of masks.
        """

    @staticmethod
    def _update_masks_executor(
        mask: Tensor, masks: list[Tensor], existing_set: set[tuple[float]], unique: bool = False
    ) -> list[Tensor]:
        """
        Add mask to masks list if not already present in existing_set.
        Updates existing_set with the new mask key if added.

        Args:
            mask: The mask tensor to potentially add.
            masks: The current list of masks.
            existing_set: Set of existing mask keys for quick lookup.
            unique: If True, only add mask if it's not already in existing_set.
        Returns:
            Updated list of masks.
        """
        key = tuple(mask.tolist())
        if not unique or (unique and key not in existing_set):
            existing_set.add(key)
            masks.append(mask)

        return masks


# pylint: disable=too-few-public-methods
class LimitedShapApproximation(BaseShapApproximation, ABC):
    """
    Base class for SHAP explainers using limited approximation methods.

    Limited stands for its limitation of not drawing same mask more than once,
    which do not align with true Monte Carlo sampling but allows for
    better coverage of feature space within limited number of samples.
    """

    def _update_masks(self, *args: Any, **kwargs: Any) -> list[Tensor]:
        kwargs["unique"] = True
        return BaseShapApproximation._update_masks_executor(*args, **kwargs)


# pylint: disable=too-few-public-methods
class StandardShapApproximation(BaseShapApproximation, ABC):
    """
    Base class for SHAP explainers using standard approximation methods.
    """

    def _update_masks(self, *args: Any, **kwargs: Any) -> list[Tensor]:
        kwargs["unique"] = False
        return BaseShapApproximation._update_masks_executor(*args, **kwargs)

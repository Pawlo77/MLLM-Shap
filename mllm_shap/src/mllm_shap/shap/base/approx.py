# pylint: disable=invalid-name
"""Base class for SHAP explainers using approximation methods."""

from abc import ABC
from logging import Logger
from typing import Any

import torch
from torch import Tensor

from ...utils.logger import get_logger
from .shap_explainer import BaseShapExplainer

logger: Logger = get_logger(__name__)


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

    include_minimal_masks: bool = True
    """Whether to include minimal masks (single-feature and empty masks) in the sampling."""

    _zero_mask_skipped: bool
    """Indicates if the zero mask was skipped."""
    _base_masks: Tensor | None
    """Holds the base masks if :attr:`include_minimal_masks` is True."""
    _base_calls_num: int
    """Number of base masks already generated."""

    def __init__(
        self,
        *args: Any,
        num_samples: int | None = None,
        fraction: float = 0.6,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            num_samples: Number of random masks to generate. If None, uses fraction.
            fraction: Fraction of total possible masks to generate if num_samples is None.
        """
        super().__init__(*args, **kwargs)
        BaseShapApproximation._validate_sampling_params(
            num_samples=num_samples,
            fraction=fraction,
        )

        self.num_samples = num_samples
        self.fraction = fraction

    def _initialize_state(self) -> None:
        """
        Initialize internal state before starting mask generation.
        """
        super()._initialize_state()

        self._zero_mask_skipped = False
        self._base_masks = None
        self._base_calls_num = 0

    def _get_next_split_base(
        self,
        n: int,
        device: torch.device,
        generated_masks_num: int,
        existing_masks: list[Tensor] | None = None,  # pylint: disable=unused-argument
    ) -> Tensor | None:
        """
        Get the next mask split for SHAP value calculation
        from the base minimal masks, if applicable.

        Args:
            n: Length of the masks
            device: Torch device to create the tensor on
            generated_masks_num: Number of masks already generated
        Returns:
            Next mask tensor or None if no more masks can be generated.
        Raises:
            RuntimeError: If there are inconsistencies in mask generation logic.
        """
        if self.include_minimal_masks:
            if generated_masks_num == 0:
                if self._first_call:
                    self._base_masks = BaseShapApproximation._generate_minimal_splits(
                        n=n,
                        device=device,
                    )
                    if self._base_masks is None:
                        return None
                    self._first_call = False
                elif not self._zero_mask_skipped:  # 0 mask was rejected, so start from 1
                    # base masks here cannot be None
                    self._base_masks = self._base_masks[1:]  # type: ignore[index]
                    self._zero_mask_skipped = True
                else:  # another mask was rejected, raise
                    raise RuntimeError("Multiple base masks were rejected.")

            if self._base_masks is None:
                raise RuntimeError("Base masks are not present.")
            num_splits = self._get_num_splits(n)
            if num_splits is not None and num_splits < self._base_masks.shape[0]:
                raise RuntimeError(
                    f"Not enough sampling budget, up to {num_splits} "
                    f"calls allowed with required {self._base_masks.shape[0]} for minimal masks."
                )

            if generated_masks_num < self._base_masks.shape[0]:
                if self._base_calls_num != generated_masks_num + int(self._zero_mask_skipped):
                    raise RuntimeError("Multiple base masks were rejected.")

                self._base_calls_num += 1
                return self._base_masks[generated_masks_num, ...].squeeze(0)
        return None

    def _get_next_split(
        self,
        n: int,
        device: torch.device,
        generated_masks_num: int,
        existing_masks: list[Tensor] | None = None,
    ) -> Tensor | None:
        r = self._get_next_split_base(
            n=n,
            device=device,
            generated_masks_num=generated_masks_num,
            existing_masks=existing_masks,
        )
        self._first_call = False
        if r is not None:  # if base mask was generated
            return r

        if generated_masks_num < self._get_num_splits(n=n):
            return self._get_random_split(n=n, device=device)
        return None

    @staticmethod
    def _generate_minimal_splits(n: int, device: torch.device) -> torch.Tensor:
        """
        Generate a minimal set of boolean masks as a batched tensor.
        Shape: (n + 1, n)
        """
        masks = torch.ones((n + 1, n), dtype=torch.bool, device=device)
        masks[0, :] = False
        masks[torch.arange(1, n + 1), torch.arange(n)] = False
        return masks

    @staticmethod
    def _get_random_split(
        n: int,
        device: torch.device,
        true_values_num: int | None = None,
        include_token: int | None = None,
    ) -> Tensor:
        """
        Generate a random split mask of shape [1, n].

        Args:
            n: Length of the mask
            device: The device to create the mask on
            true_values_num: Optional number of True values in the mask
            include_token: Optional index of a token that must be included in the mask
        Returns:
            Tensor of shape [1, n], dtype=torch.bool, representing the random split mask.
        """
        if true_values_num is None:
            return torch.randint(0, 2, (1, n), dtype=torch.bool, device=device)

        # one token is already included
        if include_token is not None:
            n -= 1
            true_values_num -= 1

        mask = torch.zeros((1, n), dtype=torch.bool, device=device)
        true_indices = torch.randperm(n, device=device)[:true_values_num]
        mask[0, true_indices] = True

        if include_token is not None:
            new_mask = torch.zeros((1, n + 1), dtype=torch.bool, device=device)
            new_mask[..., include_token] = True
            new_mask[~new_mask] = mask
            mask = new_mask
        return mask

    @staticmethod
    def _validate_sampling_params(
        num_samples: int | None,
        fraction: float | None,
    ) -> None:
        """
        Validate sampling parameters for SHAP approximation.

        Args:
            num_samples: Number of samples to generate.
            fraction: Fraction of total possible samples to generate.
        Raises:
            ValueError: If both parameters are None or invalid.
        """
        if num_samples is None and fraction is None:
            raise ValueError("Either num_samples or fraction must be provided.")
        if fraction is not None and (not isinstance(fraction, float) or not 0 < fraction <= 1):
            raise ValueError("fraction must be a float in the range (0, 1].")
        if num_samples is not None and (not isinstance(num_samples, int) or (num_samples <= 0 and num_samples != -1)):
            raise ValueError("num_samples must be a positive integer.")

"""Base class for SHAP explainers using approximation methods."""

from abc import ABC
from typing import Any

import torch
from torch import Tensor

from ...connectors.base.chat import BaseMllmChat
from ...connectors.base.model_response import ModelResponse
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

    include_minimal_masks: bool = True
    """Whether to include minimal masks (single-feature and empty masks) in the sampling."""

    _first_call: bool
    """Indicates if it's the first call to generate masks."""
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

    def _get_next_split_base(self, target_length: int, device: torch.device, generated_masks: int) -> Tensor | None:
        """
        Get the next mask split for SHAP value calculation
        from the base minimal masks, if applicable.

        Args:
            target_length: Length of the masks
            device: Torch device to create the tensor on
            generated_masks: Number of masks already generated
        Returns:
            Next mask tensor or None if no more masks can be generated.
        Raises:
            RuntimeError: If there are inconsistencies in mask generation logic.
        """
        if self.include_minimal_masks:
            if generated_masks == 0:
                if self._first_call:
                    self._base_masks = self._generate_minimal_splits(
                        target_length=target_length,
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
            num_splits = self._get_num_splits(target_length)
            if num_splits is not None and num_splits < self._base_masks.shape[0]:
                raise RuntimeError(
                    f"Not enough sampling budget, up to {num_splits} "
                    f"calls allowed with required {self._base_masks.shape[0]} for minimal masks."
                )

            if generated_masks < self._base_masks.shape[0]:
                if self._base_calls_num != generated_masks + int(self._zero_mask_skipped):
                    raise RuntimeError("Multiple base masks were rejected.")

                self._base_calls_num += 1
                return self._base_masks[generated_masks, ...]
        return None

    def _generate_minimal_splits(self, target_length: int, device: torch.device) -> torch.Tensor:
        """
        Generate a minimal set of boolean masks as a batched tensor.
        Shape: (target_length + 1, target_length).

        It ensures that masks are in following order:
        - empty mask
        - single-feature masks

        Args:
            target_length: Length of the masks
            device: Torch device to create the tensor on
        Returns:
            Batched tensor of minimal masks
        """
        return BaseShapApproximation.generate_minimal_splits(target_length, device)

    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> list[tuple[Tensor, int, BaseMllmChat | None, ModelResponse]] | None:
        self._first_call = True
        self._zero_mask_skipped = False
        self._base_masks = None
        self._base_calls_num = 0
        return super().__call__(*args, **kwargs)

    @staticmethod
    def generate_minimal_splits(target_length: int, device: torch.device) -> torch.Tensor:
        """
        Generate a minimal set of boolean masks as a batched tensor.
        Shape: (target_length + 1, target_length)
        """
        masks = torch.ones((target_length + 1, target_length), dtype=torch.bool, device=device)
        masks[0, :] = False
        masks[torch.arange(1, target_length + 1), torch.arange(target_length)] = False
        return masks

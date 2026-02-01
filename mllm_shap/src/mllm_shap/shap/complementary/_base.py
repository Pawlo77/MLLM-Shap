"""Complementary SHAP explainer implementation."""

from logging import Logger
import math
from typing import cast

import torch
from torch import Tensor

from ...utils.logger import get_logger
from ..base.complementary import BaseComplementaryShapApproximation

logger: Logger = get_logger(__name__)


# pylint: disable=too-few-public-methods
class BaseComplementaryShapExplainer(BaseComplementaryShapApproximation):
    """Complementary SHAP implementation class."""

    # pylint: disable=unused-argument,invalid-name
    def _calculate_shap_values(self, masks: Tensor, similarities: Tensor, device: torch.device) -> Tensor:
        if not self._zero_mask_skipped:
            raise RuntimeError("Zero mask was not skipped during mask generation.")
        if self._M is None:
            raise RuntimeError("M matrix must be initialized before calculating SHAP values.")

        # Adjust masks and similarities to account for skipped zero mask
        # that is remove full ones mask
        masks = masks[1:]
        similarities = similarities[1:]
        if self._C is None:
            self._calculate_C_matrix(masks=masks, similarities=similarities, device=device)

        # exclude zero-mask column
        M = self._M[:, 1:]
        C = cast(Tensor, self._C)[:, 1:]

        # it is not guaranteed especially with small budget
        non_zero_mask = M > 0
        ratio = torch.zeros_like(C)
        ratio[non_zero_mask] = C[non_zero_mask] / M[non_zero_mask]
        return torch.sum(ratio, dim=1) / M.shape[0]

    def _get_next_split_base(
        self,
        n: int,
        device: torch.device,
        generated_masks_num: int,
        existing_masks: list[Tensor] | None = None,  # pylint: disable=unused-argument
    ) -> Tensor | None:
        """
        Get the base mask for the next split.
        K-th base masks are mask with close to half of features included,
        with k-th feature definitely included. There are exactly n such masks.

        Args:
            n: Number of features.
            device: Device on which the mask should be allocated.
            generated_masks_num: Number of masks already generated.
            existing_masks: List of already generated masks (not used here).
        Returns:
            The next base mask tensor or None if all base masks have been generated.
        """
        if self.include_minimal_masks and generated_masks_num < n:
            return self._get_random_split(
                n=n,
                device=device,
                true_values_num=math.ceil(n / 3),
                include_token=generated_masks_num,
            )
        return None

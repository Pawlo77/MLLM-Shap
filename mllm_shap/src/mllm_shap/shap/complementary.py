"""Complementary SHAP explainer implementation."""

from logging import Logger
from typing import Any

import torch
from torch import Tensor

from ..utils.logger import get_logger
from .base.approx import BaseComplementaryShapApproximation

logger: Logger = get_logger(__name__)


# pylint: disable=too-few-public-methods
class ComplementaryShapExplainer(BaseComplementaryShapApproximation):
    """Complementary SHAP implementation class."""

    def _get_next_split(
        self,
        target_length: int,
        device: torch.device,
        generated_masks_num: int,
        existing_masks: list[Tensor] | None = None, # pylint: disable=unused-argument
    ) -> Tensor | None:
        self._first_call = False
        if generated_masks_num < self._get_num_splits(target_length):
            if self._next_mask is not None:
                r = self._next_mask
                self._next_mask = None
                return r

            new_mask = self._get_random_split(target_length=target_length, device=device)
            self._next_mask = ~new_mask
            return new_mask
        return None

    # pylint: disable=unused-argument,invalid-name
    def _calculate_shap_values(self, masks: Tensor, similarities: Tensor, device: torch.device) -> Tensor:
        if not self._zero_mask_skipped:
            raise RuntimeError("Zero mask was not skipped during mask generation.")        

        # Adjust masks and similarities to account for skipped zero mask
        # that is remove full ones mask
        masks = masks[1:]
        similarities = similarities[1:]
        m = masks.shape[0] // 2

        if 2 * m != masks.shape[0]:
            raise ValueError("Masks should be in complementary pairs.")

        C = torch.zeros_like(self._M, dtype=similarities.dtype, device=device)
        for i in range(m):
            if not torch.all(masks[2 * i] == ~masks[2 * i + 1]):
                raise ValueError("Masks are not complementary pairs.")

            S = masks[2 * i]
            NS = masks[2 * i + 1]
            coalition_size = int(S.sum().item())

            u = similarities[2 * i] - similarities[2 * i + 1]
            C[S, coalition_size - 1] += u
            C[NS, coalition_size - 1] -= u

        non_zero_mask = self._M != 0

        ratio = torch.zeros_like(C)
        ratio[non_zero_mask] = C[non_zero_mask] / self._M[non_zero_mask]

        return torch.sum(ratio, dim=1) / self._M.shape[0]

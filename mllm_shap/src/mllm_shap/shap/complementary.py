"""Complementary SHAP explainer implementation."""

from logging import Logger

import torch
from torch import Tensor

from ..utils.logger import get_logger
from .base.approx import BaseComplementaryShapApproximation

logger: Logger = get_logger(__name__)


# pylint: disable=too-few-public-methods
class ComplementaryShapExplainer(BaseComplementaryShapApproximation):
    """Complementary SHAP implementation class."""

    include_minimal_masks: bool = False

    # pylint: disable=unused-argument,invalid-name
    def _calculate_shap_values(self, masks: Tensor, similarities: Tensor, device: torch.device) -> Tensor:
        if not self._zero_mask_skipped:
            raise RuntimeError("Zero mask was not skipped during mask generation.")
        if self._M is None or self._C is None:
            raise RuntimeError("M and C matrices must be initialized before calculating SHAP values.")

        # Adjust masks and similarities to account for skipped zero mask
        # that is remove full ones mask
        masks = masks[1:]
        similarities = similarities[1:]
        self._calculate_C_matrix(masks=masks, similarities=similarities, device=device)

        M = self._M[:, 1:]
        C = self._C[:, 1:]

        non_zero_mask = M != 0
        ratio = torch.zeros_like(C)
        ratio[non_zero_mask] = C[non_zero_mask] / M[non_zero_mask]
        return torch.sum(ratio, dim=1) / M.shape[0]

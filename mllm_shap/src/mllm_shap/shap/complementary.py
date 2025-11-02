"""Complementary SHAP explainer implementation."""

import torch
from torch import Tensor

from .base.explainer import BaseShapExplainer


# pylint: disable=too-few-public-methods
class ComplementaryShapExplainer(BaseShapExplainer):
    """Complementary SHAP implementation generating all possible masks."""

    def _generate_masks(self, n: int, device: torch.device, existing_masks: Tensor | None = None) -> Tensor:
        raise NotImplementedError("Complementary SHAP mask generation not implemented yet.")

    # pylint: disable=too-many-locals
    def _calculate_shap_values(
        self,
        masks: Tensor,
        similarities: Tensor,
        device: torch.device,
    ) -> Tensor:
        raise NotImplementedError("Complementary SHAP value calculation is not implemented yet.")

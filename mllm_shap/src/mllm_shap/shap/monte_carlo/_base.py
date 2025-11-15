"""Base Monte Carlo approximation SHAP explainer implementation."""

from abc import ABC
from functools import lru_cache
from logging import Logger

import torch
from torch import Tensor

from ...utils.logger import get_logger
from ..base.approx import BaseShapApproximation

logger: Logger = get_logger(__name__)


# pylint: disable=too-few-public-methods
class BaseMcShapExplainer(BaseShapApproximation, ABC):
    """Base Monte Carlo SHAP implementation class"""

    @lru_cache(maxsize=1)
    def _get_num_splits(self, n: int) -> int:
        if self.num_samples is not None:
            if self.num_samples == -1:
                if self.include_minimal_masks:
                    # Minimal: only single-feature masks and empty mask
                    return n + 1
                raise ValueError("num_samples cannot be -1 when include_minimal_masks is False.")
            if self.num_samples < n + 1:
                logger.warning(
                    (
                        "Number of samples (%d) is less than number of features (%d)."
                        " Using number of features as number of samples."
                    ),
                    self.num_samples,
                    n,
                )
                return n + 1
            if self.num_samples > (2**n - 1):
                return int(2**n - 1)  # maximum possible masks excluding all-ones mask
            return self.num_samples

        total_masks = 2**n - 1  # exclude all-ones mask
        r = int(total_masks * self.fraction)
        if r < n + 1:
            r = n + 1  # minimal: single-feature masks and empty mask
            logger.warning(
                (
                    "Calculated number of samples (%d) is less than minimal"
                    " required (%d). Using minimal number of samples."
                ),
                r,
                n + 1,
            )
        return r

    # pylint: disable=unused-argument
    def _calculate_shap_values(self, masks: Tensor, similarities: Tensor, device: torch.device) -> Tensor:
        included_mean = (masks * similarities[:, None]).sum(dim=0) / masks.sum(dim=0)
        excluded_mean = ((~masks) * similarities[:, None]).sum(dim=0) / (~masks).sum(dim=0)
        return included_mean - excluded_mean

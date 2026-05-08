"""Base Monte Carlo approximation SHAP explainer implementation."""

from abc import ABC
from functools import lru_cache
from logging import Logger

import torch
from torch import Tensor

from ...utils.logger import get_logger
from ..base._mask_generator import MaskGenerator
from ..base._masks_manager import MasksManager
from ..base.approx import BaseShapApproximation
from ..core.engine import SamplingEngine
from ..core.sampling import CallableAdapterStrategy

logger: Logger = get_logger(__name__)


class BaseMcShapExplainer(BaseShapApproximation, ABC):
    """Base Monte Carlo SHAP implementation class."""

    _tqdm_desc: str = "Monte Carlo SHAP"
    """Default progress-bar label used during Monte Carlo sampling."""

    def _get_masks_generator(
        self,
        mask_manager: MasksManager,
        device: torch.device,
        masks: list[Tensor],
    ) -> MaskGenerator:
        """Create masks generator routed through SamplingEngine composition."""

        # Log Monte Carlo configuration
        probe = mask_manager._probe
        if probe:
            probe.custom_metric(
                "mc_config_num_samples",
                self.num_samples if self.num_samples is not None else -1,
            )
            probe.custom_metric("mc_config_fraction", float(self.fraction))
            probe.custom_metric(
                "mc_config_include_minimal_masks", int(self.include_minimal_masks)
            )
            probe.custom_metric(
                "mc_config_allow_duplicates", int(self.allow_mask_duplicates)
            )

            # Log calculated splits
            total_splits = self._get_num_splits(mask_manager.n)
            probe.custom_metric("mc_total_splits", total_splits)
            probe.custom_metric("mc_n_features", mask_manager.n)
            probe.custom_metric("mc_max_possible_masks", int(2**mask_manager.n - 1))

            # Log sampling strategy parameters
            probe.custom_metric("mc_expected_budget", total_splits)
            if self.num_samples is not None and self.num_samples > 0:
                probe.custom_metric("mc_budget_mode", 1)  # num_samples mode
            elif self.fraction is not None:
                probe.custom_metric("mc_budget_mode", 0)  # fraction mode

        strategy = CallableAdapterStrategy(
            get_next_split=self._get_next_split,
            get_num_splits=self._get_num_splits,
        )
        engine = SamplingEngine(
            strategy=strategy,
            allow_mask_duplicates=self.allow_mask_duplicates,
            allow_full_or_empty=False,
            probe=probe,
        )
        return engine.create_generator(
            mask_manager=mask_manager,
            device=device,
            masks=masks,
        )

    @lru_cache(maxsize=1)
    def _get_num_splits(self, n: int) -> int:
        """Return the number of Monte Carlo splits to generate."""
        if self.num_samples is not None:
            if self.num_samples == -1:
                if self.include_minimal_masks:
                    # Minimal: only single-feature masks and empty mask
                    return n + 1
                raise ValueError(
                    "num_samples cannot be -1 when include_minimal_masks is False."
                )
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

    def _calculate_shap_values(
        self, masks: Tensor, similarities: Tensor, device: torch.device
    ) -> Tensor:
        """Compute Monte Carlo SHAP values from sampled masks and similarities."""
        included_mean = (masks * similarities[:, None]).sum(dim=0) / masks.sum(dim=0)
        excluded_mean = ((~masks) * similarities[:, None]).sum(dim=0) / (~masks).sum(
            dim=0
        )
        return included_mean - excluded_mean

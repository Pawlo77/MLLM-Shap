"""Base complementary approximation shared by complementary-family explainers."""

from abc import ABC
from functools import lru_cache
from logging import Logger
from typing import Any

import torch
from torch import Tensor

from ...utils.logger import get_logger
from ..base._mask_generator import MaskGenerator
from ..base._masks_manager import MasksManager
from ..base.approx import BaseShapApproximation
from ..core.sampling import CallableAdapterStrategy
from ._engine import ComplementarySamplingEngine

logger: Logger = get_logger(__name__)


class BaseComplementaryShapApproximation(BaseShapApproximation, ABC):
    """Complementary SHAP approximation base implementation class."""

    _M: Tensor | None
    """
    Matrix M used in Complementary calculations -
    number of times feature i appears in coalitions of size j+1.
    """

    _C: Tensor | None
    """
    Matrix C used in Complementary calculations -
    C[i, j] = sum of complementary contributions for feature i in coalitions of size j+1.
    """

    @lru_cache(maxsize=1)
    def _get_num_splits(self, n: int) -> int:
        """Return the number of complementary splits to generate."""
        return BaseComplementaryShapApproximation._get_num_splits_static(
            n=n,
            num_samples=self.num_samples,
            fraction=self.fraction,
            include_minimal_masks=self.include_minimal_masks,
        )

    def _initialize_state(self) -> None:
        """Reset complementary-specific cached state before generation starts."""
        super()._initialize_state()
        self._get_num_splits.cache_clear()
        self._zero_mask_skipped = True  # this algorithm cannot use zero mask
        self._M = None
        self._C = None

    def _get_masks_generator(
        self,
        mask_manager: MasksManager,
        device: torch.device,
        masks: list[Tensor],
        allow_full_or_empty: bool = False,
    ) -> MaskGenerator:
        """Create a complementary pair generator backed by the sampling engine."""
        n = mask_manager.n

        # Initialize M matrix and log configuration
        if self._M is None:
            self._M = torch.zeros((n, n + 1), dtype=torch.int16, device=device)
        M = self._M

        # Log complementary configuration
        probe = mask_manager._probe
        if probe:
            probe.custom_metric("comp_n_features", n)
            probe.custom_metric("comp_m_matrix_shape_0", M.shape[0])
            probe.custom_metric("comp_m_matrix_shape_1", M.shape[1])
            probe.custom_metric(
                "comp_allow_duplicates", int(self.allow_mask_duplicates)
            )
            probe.custom_metric("comp_allow_full_or_empty", int(allow_full_or_empty))

            total_splits = self._get_num_splits(mask_manager.n)
            probe.custom_metric("comp_total_splits", total_splits)
            probe.custom_metric("comp_expected_pairs", total_splits // 2)

            # Log M matrix capacity
            probe.custom_metric("comp_m_matrix_total_capacity", M.shape[0] * M.shape[1])
            probe.custom_metric("comp_m_matrix_max_coalition_capacity", n + 1)

            # Log pair generation strategy
            probe.custom_metric("comp_zero_mask_skipped", int(self._zero_mask_skipped))

        # We can generate only pairs --> no space for zero mask
        # that will be a pair to existing all-ones mask
        if self._get_num_splits(mask_manager.n) % 2 == 0:
            self._zero_mask_skipped = True

        strategy = CallableAdapterStrategy(
            get_next_split=self._get_next_split,
            get_num_splits=self._get_num_splits,
        )
        engine = ComplementarySamplingEngine(
            strategy=strategy,
            allow_mask_duplicates=self.allow_mask_duplicates,
            allow_full_or_empty=allow_full_or_empty,
            probe=probe,
        )

        def _on_split_generated(split_1d: Tensor) -> None:
            """Update coalition counts for each emitted complementary split."""
            coalition_size = int(split_1d.sum().item())
            BaseComplementaryShapApproximation._increment_coalition_val(
                M,
                split_1d,
                coalition_size,
                1,
            )

        return engine.create_generator(
            mask_manager=mask_manager,
            device=device,
            masks=masks,
            on_split_generated=_on_split_generated,
        )

    def _calculate_C_matrix(
        self, masks: Tensor, similarities: Tensor, device: torch.device
    ) -> None:
        """
        Calculate the C matrix used in Complementary SHAP calculations.

        Args:
            masks: Tensor of shape [m, n] representing the generated masks.
            similarities: Tensor of shape [m, ] representing the similarities for each mask.
            device: The device to perform calculations on.
        Raises:
            ValueError: If masks are not in complementary pairs.
            RuntimeError: If M matrix is not initialized.
        """
        if self._M is None:
            raise RuntimeError(
                "M matrix must be initialized before calculating C matrix."
            )
        if self._C is None:
            self._C = torch.zeros_like(self._M, dtype=similarities.dtype, device=device)

        m = masks.shape[0] // 2
        if 2 * m != masks.shape[0]:
            raise ValueError("Masks should be in complementary pairs.")

        # Vectorized complementary pair processing
        s_masks = masks[0::2]
        ns_masks = masks[1::2]

        # Validate complementary pairing
        if not torch.all(s_masks == ~ns_masks):
            raise ValueError("Masks are not complementary pairs.")

        s_sizes = s_masks.sum(dim=1)
        ns_sizes = masks.shape[1] - s_sizes
        u = similarities[0::2] - similarities[1::2]

        for i in range(m):
            s_size = int(s_sizes[i].item())
            ns_size = int(ns_sizes[i].item())
            u_val = u[i]

            if s_size == 0:
                self._C[:, 0] += u_val
            else:
                self._C[s_masks[i], s_size] += u_val

            if ns_size == 0:
                self._C[:, 0] -= u_val
            else:
                self._C[ns_masks[i], ns_size] -= u_val

    @staticmethod
    def _get_num_splits_static(
        n: int,
        num_samples: int | None = None,
        fraction: float | None = None,
        force_minimal: bool = True,
        include_minimal_masks: bool = False,
    ) -> int:
        """Calculate the number of splits to generate for complementary SHAP approximation."""
        if num_samples is not None:
            if num_samples == -1:
                if include_minimal_masks:
                    return 2 * n
                raise ValueError(
                    "num_samples cannot be -1 when include_minimal_masks is False."
                )
            if force_minimal and num_samples < 2 * n:
                raise ValueError(
                    "num_samples must be at least equal to the number of features times two."
                )
            if num_samples > (2**n - 2):
                return int(2**n - 2)
            if num_samples % 2 == 1:
                raise ValueError(
                    "num_samples must not be odd to account for complementary masks (in pairs)."
                )
            return num_samples

        total_masks = int(2**n - 2)
        r = int(total_masks * fraction)
        if r < 2 * n:
            r = 2 * n
            logger.warning(
                (
                    "Calculated number of samples (%d) is less than "
                    "minimal required (%d). Using minimal number of samples."
                ),
                r,
                2 * n,
            )
        if r % 2 == 0:
            return r
        return r - 1

    @staticmethod
    def _increment_coalition_val(
        tensor: Tensor, indices: Tensor, coalition_size: int, value: Any
    ) -> None:
        """
        Increment the value in the tensor for the given coalition.
        If coalition_size is 0, update the first column.

        Args:
            tensor: The tensor to update.
            indices: The indices of the features in the coalition.
            coalition_size: The size of the coalition.
            value: The value to add.
        """
        if coalition_size == 0:
            tensor[:, 0] += value
        else:
            tensor[indices, coalition_size] += value

"""Complementary SHAP explainer implementation."""

from functools import lru_cache
from logging import Logger
from typing import Any

import torch
from torch import Tensor

from ..utils.logger import get_logger
from .base.approx import BaseShapApproximation
from ..connectors.base.chat import BaseMllmChat
from ..connectors.base.model_response import ModelResponse

logger: Logger = get_logger(__name__)


# pylint: disable=too-few-public-methods
class ComplementaryShapExplainer(BaseShapApproximation):
    """Complementary SHAP implementation class."""

    __next_mask: Tensor | None
    """Holds the next mask to be returned (the complement of the last generated mask)."""

    @lru_cache(maxsize=1)
    def _get_num_splits(self, target_length: int) -> int:
        if self.num_samples is not None:
            if self.num_samples < 2 * target_length:
                raise ValueError("num_samples must be at least equal to the number of features times two.")
            if self.num_samples > (2**target_length - 1):
                return int(2**target_length - 1)  # maximum possible masks excluding all-ones
            if self.num_samples % 2 == 1:
                raise ValueError("num_samples must not be odd to account for complementary masks (in pairs).")
            return self.num_samples

        # use fraction
        total_masks = int(2**target_length - 1)  # exclude all-ones mask
        r = int(total_masks * self.fraction)  # type: ignore[operator]
        # allow for odd number in this case, will add zero mask as first
        if r == total_masks:
            return r
        if r % 2 == 0:
            return r
        return r - 1  # ensure even number of samples

    def _get_next_split(self, target_length: int, device: torch.device, generated_masks: int) -> Tensor | None:
        if self._first_call:
            self._first_call = False

            if self._get_num_splits(target_length) % 2 == 1:
                # on first call, if odd number of samples, return zero mask first
                zero_mask = torch.zeros((1, target_length), dtype=torch.bool, device=device)
                return zero_mask

            # mark that zero mask was skipped
            # so that _calculate_shap_values can adjust accordingly
            self._zero_mask_skipped = True

        self._first_call = False

        if generated_masks < self._get_num_splits(target_length):
            if self.__next_mask is not None:
                r = self.__next_mask
                self.__next_mask = None
                return r

            new_mask = torch.randint(0, 2, (1, target_length), dtype=torch.bool, device=device)
            self.__next_mask = ~new_mask
            return new_mask
        return None

    # pylint: disable=unused-argument,invalid-name
    def _calculate_shap_values(self, masks: Tensor, similarities: Tensor, device: torch.device) -> Tensor:
        if self._zero_mask_skipped:
            # Adjust masks and similarities to account for skipped zero mask
            # that is remove full ones mask
            masks = masks[1:]
            similarities = similarities[1:]
            logger.debug("Adjusted masks and similarities to account for skipped zero mask.")

        n = masks.shape[1]

        M = torch.zeros((n, n), dtype=similarities.dtype, device=device)
        C = torch.zeros_like(M)

        m = masks.shape[0] // 2
        if 2 * m != masks.shape[0]:
            raise ValueError("Masks should be in complementary pairs.")

        for i in range(m):
            if not torch.all(masks[2 * i] == ~masks[2 * i + 1]):
                raise ValueError("Masks are not complementary pairs.")

            S = masks[2 * i]
            NS = masks[2 * i + 1]
            coalition_size = int(S.sum().item())

            M[S, coalition_size] += 1
            M[NS, coalition_size] -= 1

            u = similarities[2 * i] - similarities[2 * i + 1]
            C[S, coalition_size] += u
            C[NS, coalition_size] -= u

        non_zero_mask = M != 0

        ratio = torch.zeros_like(C)
        ratio[non_zero_mask] = C[non_zero_mask] / M[non_zero_mask]

        return torch.sum(ratio, dim=1) / n

    def _generate_minimal_splits(self, target_length: int, device: torch.device) -> torch.Tensor:
        """
        Generate a minimal set of boolean masks as a batched tensor.
        Shape: (2 * target_length + 1, target_length).

        It ensures that masks are in following order:
        - empty mask
        - single-feature masks and their negations interleaved

        Therefore, _calculate_shap_values can expect masks
        to be interleaved when computing SHAP values.

        Args:
            target_length: Length of the masks
            device: Torch device to create the tensor on
        Returns:
            Batched tensor of minimal masks
        """
        minimal_splits = super()._generate_minimal_splits(target_length, device)

        minimal_splits_no_zero = minimal_splits[1:]
        minimal_splits_no_zero_neg = ~minimal_splits_no_zero

        # stack along a new dimension -> (n, 2, mask_dim)
        minimal_splits_no_zero_stacked = torch.stack((minimal_splits_no_zero, minimal_splits_no_zero_neg), dim=1)
        # reshape to (2n, mask_dim) - now they are interleaved
        minimal_splits_no_zero_interleaved = minimal_splits_no_zero_stacked.reshape(-1, minimal_splits_no_zero.shape[1])

        return torch.vstack(
            [
                minimal_splits[0].unsqueeze(0),
                minimal_splits_no_zero_interleaved,
            ]
        )

    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> list[tuple[Tensor, int, BaseMllmChat | None, ModelResponse]] | None:
        self.__next_mask = None
        return super().__call__(*args, **kwargs)

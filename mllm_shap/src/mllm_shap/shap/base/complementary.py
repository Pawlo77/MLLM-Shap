# pylint: disable=invalid-name
"""Base class for SHAP explainers using approximation methods."""

from abc import ABC
from functools import lru_cache
from logging import Logger
from typing import Any, Generator

import torch
from torch import Tensor

from ...utils.logger import get_logger
from ._mask_generator import MaskGenerator
from ._masks_manager import MasksManager
from .approx import BaseShapApproximation

logger: Logger = get_logger(__name__)


# pylint: disable=too-few-public-methods
class BaseComplementaryShapApproximation(BaseShapApproximation, ABC):
    """Complementary SHAP implementation class."""

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
        return BaseComplementaryShapApproximation._get_num_splits_static(
            n=n,
            num_samples=self.num_samples,
            fraction=self.fraction,
            include_minimal_masks=self.include_minimal_masks,
        )

    def _initialize_state(self) -> None:
        super()._initialize_state()
        self._get_num_splits.cache_clear()
        self._zero_mask_skipped = True  # this algorithm cannot use zero mask
        self._M = None
        self._C = None

    # pylint: disable=too-many-arguments,too-many-positional-arguments,duplicate-code
    def _get_masks_generator(
        self,
        mask_manager: MasksManager,
        device: torch.device,
        masks: list[Tensor],
        allow_full_or_empty: bool = False,
    ) -> MaskGenerator:
        n = mask_manager.n
        num_splits = self._get_num_splits(mask_manager.n)
        get_next_split = self._get_next_split
        allow_mask_duplicates = self.allow_mask_duplicates

        # Initialize M matrix
        if self._M is None:
            self._M = torch.zeros((n, n + 1), dtype=torch.int16, device=device)
        M = self._M

        # We can generate only pairs --> no space for zero mask
        # that will be a pair to existing all-ones mask
        if self._get_num_splits(mask_manager.n) % 2 == 0:
            self._zero_mask_skipped = True

        class _MasksGenerator(MaskGenerator):
            """Generator class for masks."""

            def __init__(self) -> None:
                """Initialize the MaskGenerator."""
                super().__init__()
                self._next_result: tuple[Tensor | None, int] | None = None

            def _mask_iter(self) -> Generator[tuple[Tensor | None, int], None, None]:
                while True:
                    if self._next_result is not None:
                        yield self._next_result
                        self._next_result = None
                        continue

                    new_split = get_next_split(
                        n=mask_manager.n,
                        device=device,
                        generated_masks_num=self.generated_masks,
                        existing_masks=masks,
                    )
                    if new_split is None:
                        break

                    coalition_size = int(new_split.sum().item())
                    if not allow_full_or_empty and (not new_split.any() or new_split.all()):
                        logger.debug(
                            "Generated zero or all-ones mask of size %d, skipping.",
                            coalition_size,
                        )
                        continue

                    new_split_neg = ~new_split
                    new_mask = mask_manager.prepare_mask(split=new_split, device=device)
                    new_mask_neg = mask_manager.prepare_mask(split=new_split_neg, device=device)
                    if new_mask is None or new_mask_neg is None:
                        logger.info(
                            "Generated mask of size %d (or its negation) has no True values, skipping.",
                            coalition_size,
                        )
                        continue

                    new_mask_hash = mask_manager.get_hash(new_mask)
                    new_mask_neg_hash = mask_manager.get_hash(new_mask_neg)
                    if not allow_mask_duplicates:
                        if mask_manager.seen(mask_hash=new_mask_hash) or mask_manager.seen(mask_hash=new_mask_neg_hash):
                            logger.debug(
                                "Generated duplicate mask of size %d, skipping.",
                                coalition_size,
                            )
                            continue
                        mask_manager.mark_seen(mask_hash=new_mask_hash)
                        mask_manager.mark_seen(mask_hash=new_mask_neg_hash)

                    self.generated_masks += 2

                    for split in (new_split, new_split_neg):
                        coalition_size = int(split.sum().item())
                        logger.debug(
                            "new_split: %s, coalition_size: %d",
                            split.squeeze(0),
                            coalition_size,
                        )

                        BaseComplementaryShapApproximation._increment_coalition_val(  # pylint: disable=protected-access
                            M, split.squeeze(0), coalition_size, 1
                        )

                    self._next_result = (new_mask_neg, new_mask_neg_hash)
                    yield new_mask, new_mask_hash

            def __len__(self) -> int | None:
                return num_splits

        return _MasksGenerator()

    def _calculate_C_matrix(self, masks: Tensor, similarities: Tensor, device: torch.device) -> None:
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
            raise RuntimeError("M matrix must be initialized before calculating C matrix.")
        if self._C is None:
            self._C = torch.zeros_like(self._M, dtype=similarities.dtype, device=device)

        m = masks.shape[0] // 2
        if 2 * m != masks.shape[0]:
            raise ValueError("Masks should be in complementary pairs.")

        for i in range(m):
            if not torch.all(masks[2 * i] == ~masks[2 * i + 1]):
                raise ValueError("Masks are not complementary pairs.")

            S = masks[2 * i]
            NS = masks[2 * i + 1]
            s_size = int(S.sum().item())
            ns_size = masks.shape[1] - s_size

            u = similarities[2 * i] - similarities[2 * i + 1]

            BaseComplementaryShapApproximation._increment_coalition_val(self._C, S, s_size, u)
            BaseComplementaryShapApproximation._increment_coalition_val(self._C, NS, ns_size, -u)

    @staticmethod
    def _get_num_splits_static(
        n: int,
        num_samples: int | None = None,
        fraction: float | None = None,
        force_minimal: bool = True,
        include_minimal_masks: bool = False,
    ) -> int:
        if num_samples is not None:
            if num_samples == -1:
                if include_minimal_masks:
                    # Minimal: pairs of single-feature masks
                    return 2 * n
                raise ValueError("num_samples cannot be -1 when include_minimal_masks is False.")
            if force_minimal and num_samples < 2 * n:
                raise ValueError("num_samples must be at least equal to the number of features times two.")
            if num_samples > (2**n - 2):
                return int(2**n - 2)  # maximum possible masks excluding all-ones and all-zeros mask
            if num_samples % 2 == 1:
                raise ValueError("num_samples must not be odd to account for complementary masks (in pairs).")
            return num_samples

        # use fraction
        total_masks = int(2**n - 2)  # exclude all-ones and all-zeros mask
        r = int(total_masks * fraction)  # type: ignore[operator]
        if r < 2 * n:
            r = 2 * n  # minimal: pairs of single-feature masks
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
        return r - 1  # ensure even number of samples

    @staticmethod
    def _increment_coalition_val(tensor: Tensor, indices: Tensor, coalition_size: int, value: Any) -> None:
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

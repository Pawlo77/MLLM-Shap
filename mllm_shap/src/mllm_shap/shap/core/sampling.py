"""Sampling strategy adapters used by the composition-based sampling engines."""

from collections.abc import Callable
from logging import Logger

import torch
from torch import Tensor

from ...utils.logger import get_logger
from .contracts import SamplingStrategy

logger: Logger = get_logger(__name__)


class CallableAdapterStrategy(SamplingStrategy):
    """Adapter that bridges callable functions to SamplingStrategy contract."""

    def __init__(
        self,
        get_next_split: Callable[
            [int, torch.device, int, list[Tensor] | None],
            Tensor | None,
        ],
        get_num_splits: Callable[[int], int | None],
    ) -> None:
        """Initialize the callable-backed sampling strategy.

        Args:
            get_next_split: Callable used to generate the next split mask.
            get_num_splits: Callable used to report the expected number of splits.
        """
        self._get_next_split = get_next_split
        self._get_num_splits = get_num_splits

    def get_next_split(
        self,
        n: int,
        device: torch.device,
        generated_masks_num: int,
        existing_masks: list[Tensor] | None = None,
    ) -> Tensor | None:
        """Delegate split generation to the wrapped callable.

        Args:
            n: Number of explainable features.
            device: Device on which the split should be created.
            generated_masks_num: Number of masks generated so far.
            existing_masks: Existing masks available to the strategy.

        Returns:
            The next split mask, or ``None`` when generation should stop.
        """
        return self._get_next_split(
            n=n,
            device=device,
            generated_masks_num=generated_masks_num,
            existing_masks=existing_masks,
        )

    def get_num_splits(self, n: int) -> int | None:
        """Return the expected number of splits for ``n`` explainable features."""
        return self._get_num_splits(n)

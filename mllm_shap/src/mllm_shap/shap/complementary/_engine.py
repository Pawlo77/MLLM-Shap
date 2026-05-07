"""Pair-aware sampling engine for complementary explainers."""

from contextlib import nullcontext
from logging import Logger
from time import perf_counter
from typing import TYPE_CHECKING, Callable, Generator

import torch
from torch import Tensor

from ...utils.logger import get_logger
from ..base._mask_generator import MaskGenerator
from ..base._masks_manager import MasksManager
from ..core.contracts import SamplingStrategy
from ..core.engine import SamplingStats

if TYPE_CHECKING:
    from ..core.telemetry import TelemetryProbe

logger: Logger = get_logger(__name__)


class ComplementarySamplingEngine:
    """Pair-aware sampling engine used by complementary algorithms."""

    def __init__(
        self,
        strategy: SamplingStrategy,
        allow_mask_duplicates: bool = False,
        allow_full_or_empty: bool = False,
        probe: "TelemetryProbe | None" = None,
    ) -> None:
        """Initialize the complementary sampling engine.

        Args:
            strategy: Strategy responsible for proposing the next complementary split.
            allow_mask_duplicates: Whether duplicate mask pairs may be yielded.
            allow_full_or_empty: Whether all-zero and all-one masks may be yielded.
            probe: Optional telemetry probe used to record timing metrics.
        """
        self._strategy = strategy
        self._allow_mask_duplicates = allow_mask_duplicates
        self._allow_full_or_empty = allow_full_or_empty
        self._probe = probe

    def create_generator(
        self,
        mask_manager: MasksManager,
        device: torch.device,
        masks: list[Tensor],
        on_split_generated: Callable[[Tensor], None],
    ) -> MaskGenerator:
        """Create complementary pair generator compatible with existing SHAP flow."""
        num_splits = self._strategy.get_num_splits(mask_manager.n)
        strategy = self._strategy
        allow_mask_duplicates = self._allow_mask_duplicates
        allow_full_or_empty = self._allow_full_or_empty
        probe = self._probe if self._probe is not None and self._probe.sink else None
        stats = SamplingStats()

        class _MasksGenerator(MaskGenerator):
            """Generator class for complementary mask pairs."""

            def __init__(self) -> None:
                """Initialize the pair generator and expose shared sampling stats."""
                super().__init__()
                self.stats = stats
                self._next_result: tuple[Tensor | None, int] | None = None

            def _mask_iter(self) -> Generator[tuple[Tensor | None, int], None, None]:
                t0 = perf_counter()
                try:
                    while True:
                        if self._next_result is not None:
                            yield self._next_result
                            self._next_result = None
                            continue

                        with probe.timing("sampling") if probe else nullcontext():
                            new_split = strategy.get_next_split(
                                n=mask_manager.n,
                                device=device,
                                generated_masks_num=self.generated_masks,
                                existing_masks=masks,
                            )
                        if new_split is None:
                            break
                        self.stats.candidate_splits += 1

                        coalition_size = int(new_split.sum().item())
                        if not allow_full_or_empty and (
                            not new_split.any() or new_split.all()
                        ):
                            logger.debug(
                                "Generated zero or all-ones mask of size %d, skipping.",
                                coalition_size,
                            )
                            self.stats.skipped_full_or_empty += 1
                            continue

                        new_split_neg = ~new_split
                        with probe.timing("masking") if probe else nullcontext():
                            new_mask = mask_manager.prepare_mask(
                                split=new_split,
                                device=device,
                            )
                            new_mask_neg = mask_manager.prepare_mask(
                                split=new_split_neg,
                                device=device,
                            )
                        if new_mask is None or new_mask_neg is None:
                            logger.debug(
                                "Generated mask of size %d (or its negation) has no True values, skipping.",
                                coalition_size,
                            )
                            self.stats.skipped_invalid_masks += 1
                            continue

                        with probe.timing("dedup") if probe else nullcontext():
                            new_mask_hash = mask_manager.get_hash(new_mask)
                            new_mask_neg_hash = mask_manager.get_hash(new_mask_neg)
                            if not allow_mask_duplicates:
                                if mask_manager.seen(
                                    mask_hash=new_mask_hash
                                ) or mask_manager.seen(mask_hash=new_mask_neg_hash):
                                    logger.debug(
                                        "Generated duplicate mask of size %d, skipping.",
                                        coalition_size,
                                    )
                                    self.stats.skipped_duplicates += 1
                                    continue
                                mask_manager.mark_seen(mask_hash=new_mask_hash)
                                mask_manager.mark_seen(mask_hash=new_mask_neg_hash)

                        on_split_generated(new_split.squeeze(0))
                        on_split_generated(new_split_neg.squeeze(0))

                        self.generated_masks += 2
                        self.stats.yielded_masks += 2
                        self._next_result = (new_mask_neg, new_mask_neg_hash)
                        yield new_mask, new_mask_hash
                finally:
                    self.stats.elapsed_ms = (perf_counter() - t0) * 1000.0

            def __len__(self) -> int | None:
                """Return the expected number of complementary splits when known."""
                return num_splits

        return _MasksGenerator()

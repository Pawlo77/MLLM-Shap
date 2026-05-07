"""Sampling engines extracted from the explainer inheritance flow."""

from contextlib import nullcontext
from dataclasses import dataclass
from logging import Logger
from time import perf_counter
from typing import TYPE_CHECKING, Generator

import torch
from torch import Tensor

from ...utils.logger import get_logger
from ..base._mask_generator import MaskGenerator
from ..base._masks_manager import MasksManager
from .contracts import SamplingStrategy

if TYPE_CHECKING:
    from .telemetry import TelemetryProbe

logger: Logger = get_logger(__name__)


@dataclass
class SamplingStats:
    """Sampling runtime and filtering counters."""

    candidate_splits: int = 0
    """Number of candidate splits proposed by the sampling strategy."""

    yielded_masks: int = 0
    """Number of masks yielded to the response-generation pipeline."""

    skipped_full_or_empty: int = 0
    """Number of masks skipped because they were full or empty."""

    skipped_invalid_masks: int = 0
    """Number of masks skipped because they produced invalid prepared masks."""

    skipped_duplicates: int = 0
    """Number of masks skipped because they duplicated an already seen mask."""

    elapsed_ms: float = 0.0
    """Total generator runtime in milliseconds."""


class SamplingEngine:
    """Generic mask-generation engine driven by a sampling strategy."""

    def __init__(
        self,
        strategy: SamplingStrategy,
        allow_mask_duplicates: bool = False,
        allow_full_or_empty: bool = False,
        probe: "TelemetryProbe | None" = None,
    ) -> None:
        """Initialize the sampling engine.

        Args:
            strategy: Strategy responsible for proposing the next split.
            allow_mask_duplicates: Whether duplicate masks may be yielded.
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
    ) -> MaskGenerator:
        """Create a mask generator compatible with existing SHAP flow."""
        num_splits = self._strategy.get_num_splits(mask_manager.n)
        strategy = self._strategy
        allow_mask_duplicates = self._allow_mask_duplicates
        allow_full_or_empty = self._allow_full_or_empty
        probe = self._probe if self._probe is not None and self._probe.sink else None
        stats = SamplingStats()

        class _MasksGenerator(MaskGenerator):
            """Generator class for masks."""

            def __init__(self) -> None:
                """Initialize the mask generator and expose shared sampling stats."""
                super().__init__()
                self.stats = stats

            def _mask_iter(self) -> Generator[tuple[Tensor | None, int], None, None]:
                t0 = perf_counter()
                try:
                    while True:
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

                        if not allow_full_or_empty and (
                            not new_split.any() or new_split.all()
                        ):
                            logger.debug("Generated zero or all-ones mask, skipping.")
                            self.stats.skipped_full_or_empty += 1
                            continue

                        with probe.timing("masking") if probe else nullcontext():
                            new_mask = mask_manager.prepare_mask(
                                split=new_split,
                                device=device,
                            )
                        if new_mask is None:
                            logger.debug("Generated mask has no True values, skipping.")
                            self.stats.skipped_invalid_masks += 1
                            continue

                        with probe.timing("dedup") if probe else nullcontext():
                            new_mask_hash = mask_manager.get_hash(new_mask)
                            if not allow_mask_duplicates and mask_manager.seen(
                                mask_hash=new_mask_hash
                            ):
                                logger.debug("Generated duplicate mask, skipping.")
                                self.stats.skipped_duplicates += 1
                                continue

                            mask_manager.mark_seen(mask_hash=new_mask_hash)
                        self.generated_masks += 1
                        self.stats.yielded_masks += 1
                        yield new_mask, new_mask_hash
                finally:
                    self.stats.elapsed_ms = (perf_counter() - t0) * 1000.0

            def __len__(self) -> int | None:
                """Return the expected number of splits when it is known."""
                return num_splits

        return _MasksGenerator()

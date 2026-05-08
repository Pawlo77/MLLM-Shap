"""Sampling stage adapters over existing SHAP generation flow."""

from dataclasses import dataclass
import logging
from typing import Any, Callable

from ...base._cache_manager import CacheManager
from ...base._masks_manager import MasksManager
from ...core.telemetry import TelemetryProbe
from ..context import ExplainContext, ExplainState
from .sampling_adapter import run_sampling_generation

logger = logging.getLogger(__name__)


class InsufficientMasksError(RuntimeError):
    """Raised when no explainable masks remain after filtering."""


@dataclass(frozen=True)
class SamplingStage:
    """Stage that executes split callbacks and response generation."""

    get_next_split: Any
    """Callable returning the next sampling split/mask specification."""
    get_num_splits: Any
    """Callable returning total planned splits, if known."""
    allow_mask_duplicates: bool = False
    """Whether duplicate masks are allowed during generation."""
    allow_full_or_empty: bool = False
    """Whether fully-on/full-off masks are permitted."""
    n_generator_jobs: int = 1
    """Number of parallel jobs used for response generation."""
    progress_bar: bool = True
    """Whether to display a progress bar while generating responses."""
    verbose: bool = False
    """Whether to enable verbose logging for generation internals."""
    tqdm_desc: str = "SHAP"
    """Label shown on the progress bar."""
    generate_kwargs: dict[str, Any] | None = None
    """Optional extra keyword arguments forwarded to generation routines."""
    masks_manager_factory: Callable[..., MasksManager] = MasksManager
    """Factory used to construct the masks manager instance."""
    cache_manager_factory: Callable[..., CacheManager] = CacheManager
    """Factory used to construct the cache manager instance."""
    generate_step: Callable[..., tuple[int, Any]] | None = None
    """Optional custom generation step overriding the default adapter flow."""

    def run(
        self,
        context: ExplainContext,
        state: ExplainState,
        probe: TelemetryProbe | None = None,
    ) -> None:
        """Generate masks and model responses into pipeline state."""
        mask_manager = self.masks_manager_factory(
            chat=context.source_chat,
            log_stats=True,
            probe=probe,
        )
        cache_manager = self.cache_manager_factory(
            chat=context.response_chat,
            explainer_hash=state.metadata["explainer_hash"],
            probe=probe,
        )

        if not state.masks:
            state.masks.append(mask_manager.get_initial_mask(device=context.device))
        if not state.responses:
            state.responses.append(context.base_response)

        if self.generate_step is not None:
            skipped, history = self.generate_step(
                mask_manager=mask_manager,
                device=context.device,
                masks=state.masks,
                responses=state.responses,
                source_chat=context.source_chat,
                model=context.model,
                cache_manager=cache_manager,
                n_generator_jobs=self.n_generator_jobs,
                progress_bar=self.progress_bar,
                verbose=self.verbose,
                tqdm_desc=self.tqdm_desc,
                **(self.generate_kwargs or {}),
            )
        else:
            (skipped, history), generated_masks = run_sampling_generation(
                get_next_split=self.get_next_split,
                get_num_splits=self.get_num_splits,
                mask_manager=mask_manager,
                device=context.device,
                masks=state.masks,
                allow_mask_duplicates=self.allow_mask_duplicates,
                allow_full_or_empty=self.allow_full_or_empty,
                responses=state.responses,
                source_chat=context.source_chat,
                model=context.model,
                logger=logger,
                n_generator_jobs=self.n_generator_jobs,
                progress_bar=self.progress_bar,
                verbose=self.verbose,
                tqdm_desc=self.tqdm_desc,
                cache_manager=cache_manager,
                **(self.generate_kwargs or {}),
            )
            state.add_metadata("generated_masks", generated_masks)

        state.history = history
        state.add_metadata("chats_skipped", skipped)
        state.add_metadata("cache_extracted", cache_manager.extracted_num)

        if len(state.masks) - 1 <= skipped:
            raise InsufficientMasksError(
                "Not enough tokens to explain after filtering out empty chats. "
                "Ensure that shap_values_mask has at least two True values."
            )

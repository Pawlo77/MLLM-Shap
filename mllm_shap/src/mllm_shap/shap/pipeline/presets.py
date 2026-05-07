"""Pipeline presets for explainers."""

from dataclasses import dataclass
from typing import Any

from .pipeline import ExplainPipeline
from .stages import (
    AttributionStage,
    FinalizeStage,
    SamplingStage,
    SimilarityStage,
)


@dataclass(frozen=True)
class PipelinePreset:
    """Factory that wires explainer callbacks into composition pipeline."""

    get_next_split: Any
    """Callback that returns the next split/mask proposal."""
    get_num_splits: Any
    """Callback that reports expected number of splits, if known."""
    get_similarities: Any
    """Callback computing similarities from sampled responses."""
    get_shap_values: Any
    """Callback converting similarities into SHAP attributions."""
    save_to_cache: Any
    """Callback persisting final values into the explainer cache."""
    allow_mask_duplicates: bool
    """Whether duplicate masks are allowed in the sampling stage."""
    allow_full_or_empty: bool
    """Whether all-on/all-off masks are allowed during sampling."""
    n_generator_jobs: int
    """Number of parallel jobs used for generation."""
    progress_bar: bool
    """Whether to show progress bars in pipeline stages."""
    verbose: bool
    """Whether to enable verbose diagnostic logging."""
    tqdm_desc: str
    """Progress bar label used by generation stages."""
    generate_kwargs: dict[str, Any]
    """Extra keyword arguments forwarded into generation routines."""
    masks_manager_factory: Any | None = None
    """Optional factory overriding mask manager construction."""
    cache_manager_factory: Any | None = None
    """Optional factory overriding cache manager construction."""
    generate_step: Any | None = None
    """Optional custom generation callable replacing default sampling adapter."""
    include_sampling: bool = True
    """Whether to include the sampling stage in the built pipeline."""
    include_attribution: bool = True
    """Whether to include the attribution stage in the built pipeline."""
    include_finalize: bool = True
    """Whether to include the finalize/cache stage in the built pipeline."""

    def build(self) -> ExplainPipeline:
        """Create executable pipeline for current explainer."""
        stages: list[Any] = []

        if self.include_sampling:
            stages.append(
                SamplingStage(
                    get_next_split=self.get_next_split,
                    get_num_splits=self.get_num_splits,
                    allow_mask_duplicates=self.allow_mask_duplicates,
                    allow_full_or_empty=self.allow_full_or_empty,
                    n_generator_jobs=self.n_generator_jobs,
                    progress_bar=self.progress_bar,
                    verbose=self.verbose,
                    tqdm_desc=self.tqdm_desc,
                    generate_kwargs=self.generate_kwargs,
                    masks_manager_factory=self.masks_manager_factory,
                    cache_manager_factory=self.cache_manager_factory,
                    generate_step=self.generate_step,
                )
            )

        stages.append(SimilarityStage(get_similarities=self.get_similarities))

        if self.include_attribution:
            stages.append(AttributionStage(get_shap_values=self.get_shap_values))

        if self.include_finalize:
            stages.append(FinalizeStage(save_to_cache=self.save_to_cache))

        return ExplainPipeline(stages=tuple(stages))

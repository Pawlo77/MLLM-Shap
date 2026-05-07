"""Composition-first SHAP execution pipeline."""

from .context import ExplainContext, ExplainState
from .contracts import (
    EstimationPolicy,
    NormalizationPolicy,
    PersistencePolicy,
    PipelineStage,
    SamplingPolicy,
    SimilarityPolicy,
)
from .pipeline import ExplainPipeline
from .presets import PipelinePreset

__all__ = [
    "ExplainContext",
    "ExplainState",
    "PipelineStage",
    "SamplingPolicy",
    "SimilarityPolicy",
    "EstimationPolicy",
    "NormalizationPolicy",
    "PersistencePolicy",
    "ExplainPipeline",
    "PipelinePreset",
]

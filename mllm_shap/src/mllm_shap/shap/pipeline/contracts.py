"""Contracts for composition-first SHAP pipeline stages and policies."""

from abc import abstractmethod
from typing import Protocol

from ..core.telemetry import TelemetryProbe
from .context import ExplainContext, ExplainState


class PipelineStage(Protocol):
    """Single pipeline stage that can mutate ExplainState."""

    @abstractmethod
    def run(
        self,
        context: ExplainContext,
        state: ExplainState,
        probe: TelemetryProbe | None = None,
    ) -> None:
        """Execute stage logic and mutate state."""


class SamplingPolicy(PipelineStage, Protocol):
    """Policy producing masks and response samples."""


class SimilarityPolicy(PipelineStage, Protocol):
    """Policy computing similarities from responses."""


class EstimationPolicy(PipelineStage, Protocol):
    """Policy estimating SHAP values from masks/similarities."""


class NormalizationPolicy(PipelineStage, Protocol):
    """Policy normalizing raw SHAP values."""


class PersistencePolicy(PipelineStage, Protocol):
    """Policy persisting results (for example cache storage)."""

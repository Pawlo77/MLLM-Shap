"""Similarity stage adapters."""

from dataclasses import dataclass
from typing import Callable

from torch import Tensor

from ...core.telemetry import TelemetryProbe
from ..context import ExplainContext, ExplainState


@dataclass(frozen=True)
class SimilarityStage:
    """Stage using configured similarity callback."""

    get_similarities: Callable[..., Tensor]
    """Callback for computing similarities.
    This should be a function that takes the generated responses and the model as input
    and returns a tensor of similarity scores as output. The exact similarity metric
    used may depend on the specific SHAP variant being implemented, but common choices
    include cosine similarity or negative L2 distance between response embeddings. The
    get_similarities callback allows for flexibility in implementing different
    similarity measures, as the logic for computing similarities can be customized based
    on the requirements of the explainer and the nature of the model responses."""

    def run(
        self,
        context: ExplainContext,
        state: ExplainState,
        probe: TelemetryProbe | None = None,
    ) -> None:
        """Compute similarities from generated responses."""
        del probe
        state.similarities = self.get_similarities(
            responses=state.responses,
            model=context.model,
        )

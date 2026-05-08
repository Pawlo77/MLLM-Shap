"""Pipeline context/state for composition-first SHAP execution."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor

from ...connectors.base.chat import BaseMllmChat
from ...connectors.base.model import BaseMllmModel
from ...connectors.base.model_response import ModelResponse


@dataclass(frozen=True)
class ExplainContext:
    """Immutable input context for one explanation run."""

    model: BaseMllmModel
    """Model connector used to generate responses and embeddings."""
    source_chat: BaseMllmChat
    """Original chat instance to explain."""
    response_chat: BaseMllmChat
    """Chat instance representing the reference/response branch."""
    base_response: ModelResponse
    """Reference model response for the unmasked input."""
    device: torch.device
    """Torch device where SHAP computations are executed."""
    budget: int | None = None
    """Optional cap on the number of generated masks/samples."""
    seed: int | None = None
    """Optional random seed controlling stochastic sampling behavior."""
    params: MappingProxyType[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Immutable free-form parameters passed to pipeline stages."""


@dataclass
class ExplainState:
    """Mutable run state passed across pipeline stages."""

    masks: list[Tensor] = field(default_factory=list)
    """Accumulated masks generated across sampling stages."""
    responses: list[ModelResponse] = field(default_factory=list)
    """Model responses aligned with the generated masks."""
    similarities: Tensor | None = None
    """Similarity scores computed for current responses/masks."""
    shap_values: Tensor | None = None
    """Raw SHAP attribution values before normalization."""
    normalized_shap_values: Tensor | None = None
    """Post-processed SHAP values after normalization."""
    history: list[tuple[Tensor, int, BaseMllmChat | None, ModelResponse]] | None = None
    """Optional per-iteration generation history for debugging/analysis."""
    metadata: dict[str, Any] = field(default_factory=dict)
    """Stage-level telemetry and diagnostic metadata."""

    def add_metadata(self, key: str, value: Any) -> None:
        """Attach stage-local metadata for diagnostics/telemetry."""
        self.metadata[key] = value

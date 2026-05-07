"""Normalization and cache persistence stages."""

from dataclasses import dataclass
from typing import Callable

from ...core.telemetry import TelemetryProbe
from ..context import ExplainContext, ExplainState


@dataclass(frozen=True)
class FinalizeStage:
    """Finalize stage for normalization and cache persistence."""

    save_to_cache: Callable[..., None]
    """Callback for saving results to cache.
    This should be a function that takes the chat, source chat, responses, masks, raw SHAP values,
    and normalized SHAP values as input and saves them to the appropriate cache storage.
    The exact implementation of this callback will depend on the caching mechanism being used,
    but it should ensure that the results are stored in a way that allows for efficient retrieval and analysis later on."""

    def run(
        self,
        context: ExplainContext,
        state: ExplainState,
        probe: TelemetryProbe | None = None,
    ) -> None:
        """Persist run outputs via explainer-compatible cache callback."""
        del probe
        if state.shap_values is None:
            raise RuntimeError("shap_values missing before finalize stage")
        if state.normalized_shap_values is None:
            raise RuntimeError("normalized_shap_values missing before finalize stage")
        if "masks_tensor" not in state.metadata:
            raise RuntimeError("masks tensor metadata missing before finalize stage")

        self.save_to_cache(
            chat=context.response_chat,
            source_chat=context.source_chat,
            responses=state.responses,
            masks=state.metadata["masks_tensor"],
            shap_values=state.shap_values,
            normalized_shap_values=state.normalized_shap_values,
        )

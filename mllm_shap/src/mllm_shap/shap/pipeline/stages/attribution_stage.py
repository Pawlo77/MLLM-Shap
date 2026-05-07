"""Attribution stage adapters."""

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from ...core.telemetry import TelemetryProbe
from ..context import ExplainContext, ExplainState


@dataclass(frozen=True)
class AttributionStage:
    """Stage using high-level SHAP computation callback."""

    get_shap_values: Callable[..., tuple[Tensor, Tensor]]
    """Callback for computing SHAP values.
    This should be a function that takes the model, masks, responses, source chat, device,
    and similarities as input and returns a tuple of (raw_shap_values, normalized_shap_values) as output.
    The raw_shap_values tensor should contain the unnormalized SHAP values for each mask,
    while the normalized_shap_values tensor should contain the SHAP values normalized to sum to the difference
    between the base response and the masked response. The exact normalization method
    may depend on the specific SHAP variant being implemented. The get_shap_values callback allows
    for flexibility in implementing different SHAP estimation methods, as the logic"""

    def run(
        self,
        context: ExplainContext,
        state: ExplainState,
        probe: TelemetryProbe | None = None,
    ) -> None:
        """Compute raw and normalized SHAP values from current pipeline state."""
        del probe
        if state.similarities is None:
            raise RuntimeError("similarities missing before attribution stage")
        if not state.masks:
            raise RuntimeError("masks missing before attribution stage")
        masks_tensor = torch.stack(state.masks, dim=0)
        shap_values, normalized_shap_values = self.get_shap_values(
            model=context.model,
            masks=masks_tensor,
            responses=state.responses,
            source_chat=context.source_chat,
            device=context.device,
            similarities=state.similarities,
        )

        state.shap_values = shap_values
        state.normalized_shap_values = normalized_shap_values
        state.add_metadata("masks_tensor", masks_tensor)

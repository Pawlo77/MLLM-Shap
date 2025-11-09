"""Result model for high-level SHAP explainers."""

from pydantic import BaseModel, ConfigDict
from torch import Tensor

from ..connectors.base.chat import BaseMllmChat
from ..connectors.base.model_response import ModelResponse


# pylint: disable=too-few-public-methods
class ExplainerResult(BaseModel):
    """Result model for Explainer."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    """Configuration for pydantic model."""

    full_chat: BaseMllmChat
    """The full chat instance after generation (entire conversation). It will be set with SHAP values and cache."""

    source_chat: BaseMllmChat
    """Chat to get explained (without base response)."""

    history: list[tuple[Tensor, int, BaseMllmChat | None, ModelResponse]] | None
    """
    The history of chats and masks used during explanation
    (if applicable, that is if explainer was called with `verbose=True`).
    Each entry is a tuple of  (mask, mask_hash, masked_chat, model_response)
    If cache was used, masked_chat will be None.).
    """

    total_n_calls: int = 0
    """Total number of MLLM calls made for last explanation."""

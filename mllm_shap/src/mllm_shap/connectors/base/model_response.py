"""Model response wrapper module."""

from __future__ import annotations

from typing import TYPE_CHECKING
from pydantic import BaseModel, ConfigDict
from torch import Tensor

if TYPE_CHECKING:
    from ..base.chat import BaseMllmChat


class ModelResponse(BaseModel):
    """
    Model response wrapper.
    Used to standardize the output from different models.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    chat: BaseMllmChat | None
    """The updated chat with full history if keep_history was True."""

    generated_text_tokens: Tensor
    """The generated text tokens."""

    generated_audio_tokens: Tensor
    """The generated audio tokens."""

    generated_modality_flag: Tensor
    """The modality flag for the generated tokens."""

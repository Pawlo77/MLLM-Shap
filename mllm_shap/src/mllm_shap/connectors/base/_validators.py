"""Validators for base connectors."""

from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, field_validator

from ..config import HuggingFaceModelConfig, ModelConfig
from ..enums import ModelHistoryTrackingMode, SystemRolesSetup
from .filters import TokenFilter


class BaseChatConfig(BaseModel):
    """
    Configuration model for BaseMllmChat.
    Used just for validation and type checking.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    device: torch.device
    """The device the chat is operating on."""
    token_filter: TokenFilter
    """Token filter instance for filtering input tokens."""
    system_roles_setup: SystemRolesSetup
    """System roles setup for the chat."""
    empty_turn_sequences: set[str]
    """Set of empty turn sequences."""


class BaseModelConfig(BaseModel):
    """
    Configuration model for BaseMllmModel.
    Used just for validation and type checking.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: HuggingFaceModelConfig
    """The Hugging Face model configuration."""
    device: torch.device
    """The device the model is operating on."""
    processor: Any
    """The processor instance for the model."""
    model: Any
    """The model instance."""
    history_tracking_mode: ModelHistoryTrackingMode
    """The history tracking mode for the model."""


class BaseModelGenerateConfig(BaseModel):
    """
    Configuration model for BaseModel.generate method.
    Used just for validation and type checking.
    """

    max_new_tokens: int
    """The maximum number of new tokens to generate."""
    model_config_: ModelConfig
    """The model configuration for generation."""
    keep_history: bool
    """Whether to keep the generation history."""

    @field_validator("max_new_tokens")
    @classmethod
    def validate_max_new_tokens(cls, value: Any) -> int:
        """
        Validate max_new_tokens.

        Args:
            value: The max_new_tokens value to validate.
        Returns:
            The validated max_new_tokens value.
        Raises:
            ValueError: If max_new_tokens is not greater than 0.
        """
        parsed_value = int(value)
        if parsed_value <= 0:
            raise ValueError("max_new_tokens must be greater than 0")
        return parsed_value

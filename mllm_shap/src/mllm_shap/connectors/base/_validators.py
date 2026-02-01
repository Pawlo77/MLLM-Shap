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
    token_filter: TokenFilter
    system_roles_setup: SystemRolesSetup
    empty_turn_sequences: set[str]


# pylint: disable=duplicate-code
class BaseModelConfig(BaseModel):
    """
    Configuration model for BaseMllmModel.
    Used just for validation and type checking.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: HuggingFaceModelConfig
    device: torch.device
    processor: Any
    model: Any
    history_tracking_mode: ModelHistoryTrackingMode


class BaseModelGenerateConfig(BaseModel):
    """
    Configuration model for BaseModel.generate method.
    Used just for validation and type checking.
    """

    max_new_tokens: int
    model_config_: ModelConfig
    keep_history: bool

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

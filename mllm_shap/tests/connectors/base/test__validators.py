"""Tests for connectors.base._validators models."""

import pytest
import torch
from pydantic import ValidationError

from mllm_shap.connectors.base._validators import (
    BaseChatConfig,
    BaseModelConfig,
    BaseModelGenerateConfig,
)
from mllm_shap.connectors.config import HuggingFaceModelConfig, ModelConfig
from mllm_shap.connectors.enums import ModelHistoryTrackingMode, SystemRolesSetup
from mllm_shap.connectors.filters import KeepAllTokens


def test_base_chat_config_accepts_valid_inputs() -> None:
    """Checks that base chat config accepts valid inputs."""
    cfg = BaseChatConfig(
        device=torch.device("cpu"),
        token_filter=KeepAllTokens(),
        system_roles_setup=SystemRolesSetup.NONE,
        empty_turn_sequences={"<empty>"},
    )

    assert cfg.device.type == "cpu"
    assert cfg.system_roles_setup == SystemRolesSetup.NONE
    assert "<empty>" in cfg.empty_turn_sequences


def test_base_model_config_accepts_valid_inputs() -> None:
    """Checks that base model config accepts valid inputs."""
    cfg = BaseModelConfig(
        config=HuggingFaceModelConfig(repo_id="foo/bar", revision="main"),
        device=torch.device("cpu"),
        processor=object(),
        model=object(),
        history_tracking_mode=ModelHistoryTrackingMode.TEXT,
    )

    assert cfg.config.repo_id == "foo/bar"
    assert cfg.history_tracking_mode == ModelHistoryTrackingMode.TEXT


def test_base_model_generate_config_parses_string_max_tokens() -> None:
    """Checks that base model generate config parses string max tokens."""
    cfg = BaseModelGenerateConfig(
        max_new_tokens="7",
        model_config_=ModelConfig(),
        keep_history=True,
    )

    assert cfg.max_new_tokens == 7
    assert cfg.keep_history is True


@pytest.mark.parametrize("value", [0, -1, "0", "-2"])
def test_base_model_generate_config_rejects_non_positive_max_tokens(
    value: object,
) -> None:
    """Checks that base model generate config rejects non positive max tokens."""
    with pytest.raises(ValidationError, match="max_new_tokens must be greater than 0"):
        BaseModelGenerateConfig(
            max_new_tokens=value,
            model_config_=ModelConfig(),
            keep_history=False,
        )

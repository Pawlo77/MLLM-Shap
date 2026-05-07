"""Tests for connector token filter strategies."""

import pytest
import torch
from pydantic import ValidationError

from mllm_shap.connectors.base._validators import BaseChatConfig
from mllm_shap.connectors.filters import ExcludePunctuationTokensFilter, KeepAllTokens


def test_keep_all_tokens_has_empty_exclusion_set() -> None:
    token_filter = KeepAllTokens()

    assert token_filter.phrases_to_exclude == set()


def test_exclude_punctuation_contains_expected_symbols() -> None:
    token_filter = ExcludePunctuationTokensFilter()

    assert token_filter.phrases_to_exclude == {".", ",", "!", "?", ";", ":"}


@pytest.mark.parametrize("token", [".", ",", "!", "?", ";", ":"])
def test_exclude_punctuation_covers_each_symbol(token: str) -> None:
    token_filter = ExcludePunctuationTokensFilter()

    assert token in token_filter.phrases_to_exclude


def test_base_chat_config_accepts_token_filter_instances() -> None:
    config = BaseChatConfig(
        device=torch.device("cpu"),
        token_filter=ExcludePunctuationTokensFilter(),
        system_roles_setup=0,
        empty_turn_sequences=set(),
    )

    assert isinstance(config.token_filter, ExcludePunctuationTokensFilter)


def test_base_chat_config_rejects_invalid_token_filter_type() -> None:
    with pytest.raises(ValidationError):
        BaseChatConfig(
            device=torch.device("cpu"),
            token_filter="invalid",
            system_roles_setup=0,
            empty_turn_sequences=set(),
        )

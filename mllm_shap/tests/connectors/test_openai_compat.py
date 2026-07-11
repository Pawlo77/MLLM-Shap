"""Tests for OpenAI-compatible HTTP text connector (mocked transport)."""

from __future__ import annotations

from typing import Any, Mapping

import pytest
import torch

from mllm_shap.connectors.enums import Role
from mllm_shap.connectors.openai_compat import OpenAICompatCausalText
from mllm_shap.connectors.config import ModelConfig


def _fake_transport(
    url: str, headers: Mapping[str, str], payload: Mapping[str, Any]
) -> dict[str, Any]:
    assert "/chat/completions" in url
    assert payload["model"]
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hi there",
                }
            }
        ]
    }


@pytest.mark.parametrize("device", [torch.device("cpu")])
def test_openai_compat_generate_mocked(device: torch.device) -> None:
    m = OpenAICompatCausalText(
        device,
        base_url="http://test.invalid/v1",
        chat_model="dummy",
        transport=_fake_transport,
    )
    chat = m.get_new_chat()
    chat.new_turn(Role.USER)
    chat.add_text("Hello")
    chat.end_turn()
    resp = m.generate(
        chat,
        max_new_tokens=8,
        model_config=ModelConfig(text_temperature=0.0),
        keep_history=False,
    )
    assert resp.generated_text_tokens.numel() > 0


@pytest.mark.parametrize("device", [torch.device("cpu")])
def test_openai_compat_static_embeddings_shape(device: torch.device) -> None:
    m = OpenAICompatCausalText(
        device,
        base_url="http://test.invalid/v1",
        chat_model="dummy",
        transport=_fake_transport,
    )
    chat = m.get_new_chat()
    chat.new_turn(Role.USER)
    chat.add_text("x")
    chat.end_turn()
    resp = m.generate(
        chat,
        max_new_tokens=4,
        model_config=ModelConfig(text_temperature=0.0),
        keep_history=False,
    )
    embs = m.get_static_embeddings([resp])
    assert len(embs) == 1
    assert embs[0].dim() == 2
    ctx = m._get_contextual_embeddings(embs)
    assert len(ctx) == 1
    assert ctx[0].shape == embs[0].shape

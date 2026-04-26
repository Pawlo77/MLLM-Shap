"""Tests for TransformersCausalText model connector."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import Tensor

from mllm_shap.connectors.config import ModelConfig
from mllm_shap.connectors.enums import ModelHistoryTrackingMode
from mllm_shap.connectors.transformers_text import model as text_model


class _TokenizerStub:
    """Tokenizer stub used by connector tests."""

    pad_token_id: int | None = 0
    eos_token_id: int | None = 2
    eos_token: str | None = "</s>"
    pad_token: str | None = None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(ch) % 19 for ch in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return ",".join(str(i) for i in ids)


class _BaseModelStub:
    """Base model stub for contextual embeddings."""

    def __call__(self, *, inputs_embeds: Tensor, use_cache: bool) -> SimpleNamespace:
        del use_cache
        return SimpleNamespace(last_hidden_state=inputs_embeds + 1)


class _ModelStub:
    """HF causal model stub."""

    def __init__(self) -> None:
        self.generation_config = text_model.GenerationConfig()
        self.generate_calls: list[dict[str, Any]] = []
        self.base_model = _BaseModelStub()

    def to(self, device: torch.device) -> "_ModelStub":
        del device
        return self

    def eval(self) -> "_ModelStub":
        return self

    def generate(self, **kwargs: Any) -> SimpleNamespace:
        self.generate_calls.append(kwargs)
        input_ids: Tensor = kwargs["input_ids"]
        seq = torch.cat([input_ids, torch.tensor([[5, 6]], dtype=torch.long)], dim=1)
        return SimpleNamespace(sequences=seq)

    def get_input_embeddings(self) -> Any:
        def embed(ids: Tensor) -> Tensor:
            # [1, T] -> [1, T, 1]
            return ids.to(dtype=torch.float32).unsqueeze(-1)

        return embed


@pytest.fixture
def patched_connector(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Patch heavy HF loading calls with light stubs."""

    def fake_tok_from_pretrained(*args: Any, **kwargs: Any) -> _TokenizerStub:
        del args, kwargs
        return _TokenizerStub()

    def fake_model_from_pretrained(*args: Any, **kwargs: Any) -> _ModelStub:
        del args, kwargs
        return _ModelStub()

    monkeypatch.setattr(
        text_model.AutoTokenizer, "from_pretrained", fake_tok_from_pretrained
    )
    monkeypatch.setattr(
        text_model.AutoModelForCausalLM, "from_pretrained", fake_model_from_pretrained
    )
    return text_model.TransformersCausalText(device=torch.device("cpu"))


def test_init_rejects_forbidden_kwargs() -> None:
    """Explicit model/processor/config should be rejected."""
    with pytest.raises(
        ValueError, match="Do not pass 'config', 'model', or 'processor'"
    ):
        _ = text_model.TransformersCausalText(
            device=torch.device("cpu"), model=object()
        )


def test_init_forces_text_history_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-TEXT history mode should be forced to TEXT with warning."""

    monkeypatch.setattr(
        text_model.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: _TokenizerStub(),
    )
    monkeypatch.setattr(
        text_model.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: _ModelStub(),
    )
    with pytest.warns(UserWarning, match="Forcing TEXT mode"):
        model = text_model.TransformersCausalText(
            device=torch.device("cpu"),
            history_tracking_mode=ModelHistoryTrackingMode.AUDIO,
        )
    assert model.history_tracking_mode == ModelHistoryTrackingMode.TEXT


def test_generate_text_response_and_history(
    patched_connector: text_model.TransformersCausalText,
) -> None:
    """Generate should return text-only tokens and optionally keep history."""
    chat = patched_connector.get_new_chat()
    chat._add_text("ab")
    response = patched_connector.generate(
        chat=chat, max_new_tokens=2, model_config=ModelConfig(), keep_history=True
    )
    assert response.generated_text_tokens.tolist() == [5, 6]
    assert response.generated_audio_tokens.shape == (0, 0)
    assert response.generated_modality_flag.shape[0] == 2
    assert response.chat is not None


def test_generate_warns_when_audio_knobs_passed(
    patched_connector: text_model.TransformersCausalText,
) -> None:
    """Audio config knobs should trigger warning in text connector."""
    chat = patched_connector.get_new_chat()
    chat._add_text("ab")
    with pytest.warns(UserWarning, match="audio settings are ignored"):
        _ = patched_connector.generate(
            chat=chat,
            model_config=ModelConfig(audio_temperature=0.5, audio_top_k=4),
        )


def test_embeddings_api(
    patched_connector: text_model.TransformersCausalText,
) -> None:
    """Static/contextual embeddings should preserve token count."""
    responses = [
        text_model.ModelResponse(
            chat=None,
            generated_text_tokens=torch.tensor([1, 2, 3], dtype=torch.long),
            generated_audio_tokens=torch.empty((0, 0), dtype=torch.long),
            generated_modality_flag=torch.tensor([0, 0, 0], dtype=torch.long),
        )
    ]
    static = patched_connector.get_static_embeddings(responses)
    contextual = patched_connector.get_contextual_embeddings(static_embeddings=static)
    assert len(static) == 1 and static[0].shape[:1] == (3,)
    assert len(contextual) == 1 and contextual[0].shape[:1] == (3,)

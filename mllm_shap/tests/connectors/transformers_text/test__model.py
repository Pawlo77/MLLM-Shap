"""Tests for TransformersCausalText model connector."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import Tensor

from mllm_shap.connectors.config import ModelConfig
from mllm_shap.connectors.enums import ModelHistoryTrackingMode
from mllm_shap.connectors.transformers_text import model as text_model
from .conftest import _TokenizerStub


class _BaseModelStub:
    """Base model stub for contextual embeddings."""

    def __call__(self, inputs_embeds: Tensor, use_cache: bool) -> SimpleNamespace:
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


class _TokenizerPadFallbackStub(_TokenizerStub):
    """Tokenizer stub with missing pad token id to trigger pad fallback path."""

    pad_token_id: int | None = None
    eos_token_id: int | None = 2
    eos_token: str | None = "</s>"
    pad_token: str | None = None


class _ModelNonGenerationConfigStub(_ModelStub):
    """Model stub with non-GenerationConfig object to trigger fallback creation."""

    def __init__(self) -> None:
        super().__init__()
        self.generation_config = SimpleNamespace(pad_token_id=None, eos_token_id=None)


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
        chat=chat,
        max_new_tokens=2,
        model_config=ModelConfig(audio_temperature=None, audio_top_k=None),
        keep_history=True,
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


def test_init_sets_pad_token_from_eos_when_pad_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing tokenizer pad token id should fall back to EOS token."""
    tokenizer = _TokenizerPadFallbackStub()

    monkeypatch.setattr(
        text_model.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )
    monkeypatch.setattr(
        text_model.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: _ModelStub(),
    )

    _ = text_model.TransformersCausalText(device=torch.device("cpu"))
    assert tokenizer.pad_token == tokenizer.eos_token


def test_init_builds_generation_config_when_model_has_non_standard_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-GenerationConfig model config should be replaced and ids initialized."""
    tokenizer = _TokenizerStub()
    model = _ModelNonGenerationConfigStub()

    monkeypatch.setattr(
        text_model.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )
    monkeypatch.setattr(
        text_model.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: model,
    )

    connector = text_model.TransformersCausalText(device=torch.device("cpu"))
    assert isinstance(connector.model.generation_config, text_model.GenerationConfig)
    assert connector.model.generation_config.pad_token_id == tokenizer.pad_token_id
    assert connector.model.generation_config.eos_token_id == tokenizer.eos_token_id


def test_init_keeps_existing_generation_config_ids_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing generation config ids should not be overwritten."""
    tokenizer = _TokenizerStub()
    model = _ModelStub()
    model.generation_config.pad_token_id = 123
    model.generation_config.eos_token_id = 456

    monkeypatch.setattr(
        text_model.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )
    monkeypatch.setattr(
        text_model.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: model,
    )

    connector = text_model.TransformersCausalText(device=torch.device("cpu"))
    assert connector.model.generation_config.pad_token_id == 123
    assert connector.model.generation_config.eos_token_id == 456


def test_contextual_embeddings_accept_rank3_static_embeddings(
    patched_connector: text_model.TransformersCausalText,
) -> None:
    """Rank-3 static embeddings should bypass unsqueeze branch and still work."""
    static = [torch.ones((1, 3, 2), dtype=torch.float32)]
    contextual = patched_connector.get_contextual_embeddings(static_embeddings=static)
    assert len(contextual) == 1
    assert contextual[0].shape == (3, 2)

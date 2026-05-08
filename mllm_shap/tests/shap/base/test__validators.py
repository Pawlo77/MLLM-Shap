"""Tests for shap.base._validators models."""

import pytest
from pydantic import ValidationError

from mllm_shap.connectors.base.model_response import ModelResponse
from mllm_shap.shap.base._validators import BaseShapCallConfig, BaseShapConfig
from mllm_shap.shap.embeddings import MeanReducer
from mllm_shap.shap.enums import Mode
from mllm_shap.shap.normalizers import IdentityNormalizer
from mllm_shap.shap.similarity import CosineSimilarity

from ...dummy import DummyChat, DummyModel


def test_base_shap_config_accepts_valid_inputs() -> None:
    """Checks that base shap config accepts valid inputs."""
    cfg = BaseShapConfig(
        mode=Mode.STATIC,
        embedding_model=None,
        embedding_reducer=MeanReducer(),
        similarity_measure=CosineSimilarity(),
        normalizer=IdentityNormalizer(),
        allow_mask_duplicates=False,
    )

    assert cfg.mode == Mode.STATIC
    assert cfg.allow_mask_duplicates is False


def test_base_shap_call_config_rejects_mismatched_chat_device() -> None:
    """Checks that base shap call config rejects mismatched chat device."""
    model = DummyModel()
    source_chat = DummyChat()
    response_chat = DummyChat()
    response_chat.device = "cuda:0"

    response = ModelResponse(
        chat=response_chat,
        generated_text_tokens=source_chat.input_tokens,
        generated_audio_tokens=source_chat.audio_tokens,
        generated_modality_flag=source_chat.tokens_modality_flag,
    )

    with pytest.raises(ValidationError, match="same device"):
        BaseShapCallConfig(
            model=model,
            source_chat=source_chat,
            response=response,
            progress_bar=False,
            verbose=False,
        )


def test_base_shap_call_config_rejects_missing_response_chat() -> None:
    """Checks that base shap call config rejects missing response chat."""
    model = DummyModel()
    source_chat = DummyChat()

    response = ModelResponse(
        chat=source_chat,
        generated_text_tokens=source_chat.input_tokens,
        generated_audio_tokens=source_chat.audio_tokens,
        generated_modality_flag=source_chat.tokens_modality_flag,
    )
    response.chat = None

    with pytest.raises(ValidationError, match="Response must have a chat instance"):
        BaseShapCallConfig(
            model=model,
            source_chat=source_chat,
            response=response,
            progress_bar=False,
            verbose=False,
        )

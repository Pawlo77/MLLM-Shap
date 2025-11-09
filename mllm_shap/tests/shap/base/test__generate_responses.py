"""Unit tests for generate_responses function and _process_mask helper."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from mllm_shap.connectors.base.chat import AllTextTokensFilteredOutError
from mllm_shap.connectors.base.model_response import ModelResponse
from mllm_shap.shap.base._cache_manager import CacheManager
from mllm_shap.shap.base._generate_responses import (
    _process_mask,
    generate_responses,
)
from torch import Tensor

from ...dummy import DummyChat, DummyModel


class TestGenerateResponses:
    """Unit tests for generate_responses and _process_mask behavior."""

    @pytest.fixture(scope="function")
    def setup_env(self) -> tuple[DummyModel, DummyChat, CacheManager]:
        """Fixture providing dummy model, chat, and cache manager."""
        model = DummyModel()
        chat = DummyChat()
        cache_manager = CacheManager(chat=chat, explainer_hash=123)
        return model, chat, cache_manager

    @patch("mllm_shap.shap.base.explainer.CacheManager")
    def test_process_mask_from_cache(
        self,
        cache_manager: MagicMock,
        setup_env: tuple[DummyModel, DummyChat, CacheManager],
    ) -> None:
        """Should return cached response if mask is already in cache."""
        model, chat, _ = setup_env
        mask = torch.tensor([True, False, True])
        mask_hash = hash(tuple(mask.tolist()))
        cached_model_response = ModelResponse(
            chat=chat,
            generated_text_tokens=torch.tensor([1, 2, 3]),
            generated_audio_tokens=torch.tensor([]),
            generated_modality_flag=torch.ones(3, dtype=torch.bool),
        )

        mock_cache_manager = cache_manager.return_value
        mock_cache_manager.contains.return_value = True
        mock_cache_manager.extract.return_value = cached_model_response

        masked_chat, model_response = _process_mask(
            mask=mask,
            mask_hash=mask_hash,
            source_chat=chat,
            model=model,
            cache_manager=mock_cache_manager,
            verbose=False,
            i=0,
        )

        assert masked_chat is None
        mock_cache_manager.extract.assert_called_once()
        assert model_response is cached_model_response

    def test_process_mask_generate_new(self, setup_env: tuple[DummyModel, DummyChat, CacheManager]) -> None:
        """Should generate new response and cache it if not found."""
        model, chat, cache_manager = setup_env
        mask = torch.tensor([True, True, True])
        mask_hash = hash(tuple(mask.tolist()))

        masked_chat, model_response = _process_mask(
            mask=mask,
            mask_hash=mask_hash,
            source_chat=chat,
            model=model,
            cache_manager=cache_manager,
            verbose=True,
            i=1,
        )

        assert isinstance(masked_chat, DummyChat)
        assert model_response is not None
        assert hasattr(model_response, "generated_text_tokens")

    def test_process_mask_raises_all_text_filtered(
        self,
        setup_env: tuple[DummyModel, DummyChat, CacheManager],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Should raise AllTextTokensFilteredOutError when chat filter removes all tokens."""
        model, chat, cache_manager = setup_env

        # Patch DummyChat.from_chat temporarily to simulate raising
        def raise_all_text_filtered(mask, chat):
            raise AllTextTokensFilteredOutError()

        monkeypatch.setattr(
            "mllm_shap.connectors.base.chat.BaseMllmChat.from_chat",
            staticmethod(raise_all_text_filtered),
        )

        mask = torch.tensor([False, False, False])
        mask_hash = hash(tuple(mask.tolist()))

        with pytest.raises(AllTextTokensFilteredOutError):
            _process_mask(
                mask=mask,
                mask_hash=mask_hash,
                source_chat=chat,
                model=model,
                cache_manager=cache_manager,
                verbose=False,
                i=2,
            )

    def test_generate_responses_single(self, setup_env: tuple[DummyModel, DummyChat, CacheManager]) -> None:
        """Should correctly generate responses in single-threaded mode."""
        model, chat, cache_manager = setup_env

        gen = ((torch.tensor([True, False, True]), hash(b"mask1")),)
        masks, responses = [], []

        chats_skipped, history = generate_responses(
            masks=masks,
            responses=responses,
            gen=gen,
            source_chat=chat,
            model=model,
            cache_manager=cache_manager,
            n_generator_jobs=1,
            progress_bar=False,
            verbose=True,
        )

        assert chats_skipped == 0
        assert len(masks) == 1
        assert len(responses) == 1
        assert isinstance(history[0][3].generated_text_tokens, Tensor)

    def test_generate_responses_multi(self, setup_env: tuple[DummyModel, DummyChat, CacheManager]) -> None:
        """Should correctly generate responses using multiple generator jobs."""
        model, chat, cache_manager = setup_env

        gen = (
            (torch.tensor([True, False, True]), hash(b"mask1")),
            (torch.tensor([True, True, True]), hash(b"mask2")),
        )
        masks, responses = [], []

        chats_skipped, history = generate_responses(
            masks=masks,
            responses=responses,
            gen=gen,
            source_chat=chat,
            model=model,
            cache_manager=cache_manager,
            n_generator_jobs=2,
            progress_bar=False,
            verbose=True,
        )

        assert chats_skipped == 0
        assert len(masks) == 2
        assert len(responses) == 2
        assert isinstance(history[0][3].generated_text_tokens, Tensor)

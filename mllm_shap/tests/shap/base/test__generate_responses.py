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

    @patch("mllm_shap.shap.base._generate_responses.CacheManager")
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

    def test_process_mask_generate_new(
        self, setup_env: tuple[DummyModel, DummyChat, CacheManager]
    ) -> None:
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

    def test_generate_responses_single(
        self, setup_env: tuple[DummyModel, DummyChat, CacheManager]
    ) -> None:
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

    def test_generate_responses_multi(
        self, setup_env: tuple[DummyModel, DummyChat, CacheManager]
    ) -> None:
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

    def test_process_mask_respects_keep_history(
        self,
        setup_env: tuple[DummyModel, DummyChat, CacheManager],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Model.generate should receive keep_history matching verbose flag."""
        model, chat, cache_manager = setup_env
        calls: list[bool] = []

        def fake_generate(
            self, chat: DummyChat, keep_history: bool = False, **kwargs
        ) -> ModelResponse:  # noqa: D401
            del chat, kwargs
            calls.append(keep_history)
            return ModelResponse(
                chat=DummyChat(),
                generated_text_tokens=torch.tensor([1, 2]),
                generated_audio_tokens=torch.tensor([]),
                generated_modality_flag=torch.ones(2, dtype=torch.bool),
            )

        monkeypatch.setattr(DummyModel, "generate", fake_generate)

        for verbose in (False, True):
            _process_mask(
                mask=torch.tensor([True, True, True]),
                mask_hash=123,
                source_chat=chat,
                model=model,
                cache_manager=cache_manager,
                verbose=verbose,
                i=0,
            )

        assert calls == [False, True]

    @patch("mllm_shap.shap.base._generate_responses._process_mask")
    def test_generate_responses_single_skips_and_collects_history(
        self,
        mock_process: MagicMock,
        setup_env: tuple[DummyModel, DummyChat, CacheManager],
    ) -> None:
        """Single-threaded flow should count skipped masks and keep history when verbose."""
        model, chat, cache_manager = setup_env

        base_response = ModelResponse(
            chat=chat,
            generated_text_tokens=torch.tensor([1, 2, 3]),
            generated_audio_tokens=torch.tensor([]),
            generated_modality_flag=torch.ones(3, dtype=torch.bool),
        )

        def side_effect(*, verbose: bool, **kwargs):
            if side_effect.calls == 0:
                side_effect.calls += 1
                raise AllTextTokensFilteredOutError()
            side_effect.calls += 1
            return DummyChat(), base_response

        side_effect.calls = 0
        mock_process.side_effect = side_effect

        gen = iter(
            [
                (torch.tensor([True, False, False]), 1),
                (torch.tensor([True, True, False]), 2),
            ]
        )
        masks: list[Tensor] = []
        responses: list[ModelResponse] = []

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

        assert chats_skipped == 1
        assert len(masks) == 1
        assert len(responses) == 1
        assert history is not None and len(history) == 1
        assert mock_process.call_count == 2

    def test_generate_responses_single_history_none_when_not_verbose(
        self,
        setup_env: tuple[DummyModel, DummyChat, CacheManager],
    ) -> None:
        """History should be omitted when verbose flag is False."""
        model, chat, cache_manager = setup_env
        gen = iter([(torch.tensor([True, True, True]), 1)])
        masks: list[Tensor] = []
        responses: list[ModelResponse] = []

        chats_skipped, history = generate_responses(
            masks=masks,
            responses=responses,
            gen=gen,
            source_chat=chat,
            model=model,
            cache_manager=cache_manager,
            n_generator_jobs=1,
            progress_bar=False,
            verbose=False,
        )

        assert chats_skipped == 0
        assert history is None

    @patch("mllm_shap.shap.base._generate_responses._process_mask")
    def test_generate_responses_multi_propagates_verbose_flag(
        self,
        mock_process: MagicMock,
        setup_env: tuple[DummyModel, DummyChat, CacheManager],
    ) -> None:
        """All worker invocations should inherit verbose parameter."""
        model, chat, cache_manager = setup_env
        verbose_flags: list[bool] = []

        def worker_side_effect(*, verbose: bool, **kwargs):
            verbose_flags.append(verbose)
            return DummyChat(), ModelResponse(
                chat=chat,
                generated_text_tokens=torch.tensor([1]),
                generated_audio_tokens=torch.tensor([]),
                generated_modality_flag=torch.ones(1, dtype=torch.bool),
            )

        mock_process.side_effect = worker_side_effect

        gen = iter(
            [
                (torch.tensor([True, False, True]), 1),
                (torch.tensor([False, True, True]), 2),
            ]
        )
        masks: list[Tensor] = []
        responses: list[ModelResponse] = []

        _, history = generate_responses(
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

        assert verbose_flags == [True, True]
        assert history is not None and len(history) == 2

    @patch("mllm_shap.shap.base._generate_responses._process_mask")
    def test_generate_responses_multi_raises_on_worker_error(
        self,
        mock_process: MagicMock,
        setup_env: tuple[DummyModel, DummyChat, CacheManager],
    ) -> None:
        """Unhandled worker exception should surface as RuntimeError."""
        model, chat, cache_manager = setup_env
        mock_process.side_effect = RuntimeError("boom")

        gen = iter([(torch.tensor([True, False, True]), 1)])

        with pytest.raises(RuntimeError, match="worker"):
            generate_responses(
                masks=[],
                responses=[],
                gen=gen,
                source_chat=chat,
                model=model,
                cache_manager=cache_manager,
                n_generator_jobs=2,
                progress_bar=False,
                verbose=True,
            )

    @patch("mllm_shap.shap.base._generate_responses.CacheManager")
    def test_process_mask_uses_mask_when_hash_missing(
        self,
        cache_manager: MagicMock,
        setup_env: tuple[DummyModel, DummyChat, CacheManager],
    ) -> None:
        """Missing mask_hash should fall back to mask-based lookup."""
        model, chat, _ = setup_env
        mask = torch.tensor([True, False, True])
        mock_cache = cache_manager.return_value
        mock_cache.contains.return_value = True
        mock_cache.extract.return_value = ModelResponse(
            chat=chat,
            generated_text_tokens=torch.tensor([1]),
            generated_audio_tokens=torch.tensor([]),
            generated_modality_flag=torch.ones(1, dtype=torch.bool),
        )

        _process_mask(
            mask=mask,
            mask_hash=None,
            source_chat=chat,
            model=model,
            cache_manager=mock_cache,
            verbose=False,
            i=0,
        )

        mock_cache.contains.assert_called_once_with(mask_hash=None)
        mock_cache.extract.assert_called_once_with(mask_hash=None)

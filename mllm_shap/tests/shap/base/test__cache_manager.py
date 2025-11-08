"""Unit tests for CacheManager class."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from mllm_shap.connectors.base.chat import BaseMllmChat
from mllm_shap.connectors.base.explainer_cache import ExplainerCache
from mllm_shap.connectors.base.model_response import ModelResponse
from mllm_shap.shap.base._cache_manager import CacheManager

from ...dummy import DummyChat


class TestCacheManager:
    """Unit tests for CacheManager methods and validation."""

    @pytest.fixture
    def chat(self) -> BaseMllmChat:
        """Fixture for a mock chat object."""
        return DummyChat(num_tokens=5)

    @pytest.fixture
    def cache(self, chat: BaseMllmChat) -> ExplainerCache:
        """Fixture for ExplainerCache instance."""
        responses = [
            ModelResponse(
                chat=None,
                generated_audio_tokens=torch.zeros(1, 2),
                generated_modality_flag=torch.zeros(1, 2),
                generated_text_tokens=torch.zeros(1, 2),
            )
            for _ in range(2)
        ]
        masks = torch.tensor(
            [[True, False, True, False, False], [False, True, False, False, False]],
            dtype=torch.bool,
        )
        return ExplainerCache(
            chat=chat,
            calculated_by=123,
            n=5,
            responses=responses,
            masks=masks,
        )

    @patch("mllm_shap.shap.base._cache_manager.MasksManager")
    def test_init_creates_mask_manager_and_removes_cache(
        self, mock_masks_manager: MagicMock, chat: BaseMllmChat
    ) -> None:
        """Should initialize mask manager and set chat.cache=None."""
        chat.cache = None
        manager = CacheManager(chat=chat, explainer_hash=123)
        assert manager._masks_manager is mock_masks_manager.return_value
        assert manager.cache is None
        assert chat.cache is None
        assert manager._responses_map == {}

    @patch("mllm_shap.shap.base._cache_manager.MasksManager")
    def test_init_with_valid_existing_cache(
        self, mock_masks_manager: MagicMock, chat: BaseMllmChat, cache: ExplainerCache
    ) -> None:
        """Should correctly attach and process existing cache."""
        masks_manager = mock_masks_manager.return_value
        masks_manager.get_hash.side_effect = lambda mask: hash(mask.numpy().tobytes())
        masks_manager.mark_seen.side_effect = lambda mask_hash: None

        chat.cache = cache
        manager = CacheManager(chat=chat, explainer_hash=123)
        assert manager.cache == cache
        # all masks processed and marked
        for mask in cache.masks:
            h = masks_manager.get_hash(mask)
            masks_manager.mark_seen.assert_any_call(mask_hash=h)
            assert h in manager._responses_map
        # chat.cache must be cleared
        assert chat.cache is None

    def test_init_raises_if_different_explainer_hash(self, chat: BaseMllmChat, cache: ExplainerCache) -> None:
        """Should raise if cache calculated_by != explainer_hash."""
        chat.cache = cache
        with pytest.raises(ValueError, match="different explainer instance"):
            _ = CacheManager(chat=chat, explainer_hash=999)

    def test_init_raises_if_different_chat_instance(self, chat: BaseMllmChat, cache: ExplainerCache) -> None:
        """Should raise if cache.chat != current chat."""
        another_chat = MagicMock()
        cache.chat = another_chat
        chat.cache = cache
        with pytest.raises(ValueError, match="different chat instance"):
            _ = CacheManager(chat=chat, explainer_hash=123)

    @patch("mllm_shap.shap.base._cache_manager.MasksManager")
    def test_contains_returns_true_if_seen(self, mock_mask_manager: MagicMock, chat: BaseMllmChat) -> None:
        """contains() should return True if mask has been seen."""
        mock_manager = mock_mask_manager.return_value
        mock_manager.seen.return_value = True

        manager = CacheManager(chat=chat, explainer_hash=123)
        result = manager.contains(mask=torch.tensor([True, False, True, False, False]))

        assert result is True
        mock_manager.seen.assert_called_once()

    def test_extract_returns_correct_response(self, chat: BaseMllmChat, cache: ExplainerCache) -> None:
        """extract() should return the correct cached response."""
        chat.cache = cache
        manager = CacheManager(chat=chat, explainer_hash=123)

        # manually set mapping for test
        mask = cache.masks[0]
        mask_hash = manager._masks_manager.get_hash(mask)
        manager._responses_map[mask_hash] = 0
        manager.cache = cache

        result = manager.extract(mask=mask)
        assert isinstance(result, ModelResponse)
        assert result is cache.responses[0]
        assert manager.extracted_num == 1

    def test_extract_raises_if_no_cache(self, chat: BaseMllmChat) -> None:
        """Should raise if no cache is attached."""
        manager = CacheManager(chat=chat, explainer_hash=123)
        manager.cache = None
        with pytest.raises(ValueError, match="No cache is associated"):
            _ = manager.extract(mask=torch.tensor([True, False, True, False, False]))

    def test_extract_raises_if_mask_not_found(self, chat: BaseMllmChat, cache: ExplainerCache) -> None:
        """Should raise if mask not present in responses map."""
        chat.cache = cache
        manager = CacheManager(chat=chat, explainer_hash=123)
        manager.cache = cache
        manager._responses_map = {}
        mask = cache.masks[0]
        with pytest.raises(KeyError, match="Mask not found"):
            _ = manager.extract(mask=mask)

    def test_extract_raises_if_no_mask_or_hash(self, chat: BaseMllmChat, cache: ExplainerCache) -> None:
        """Should raise if both mask and mask_hash are missing."""
        chat.cache = cache
        manager = CacheManager(chat=chat, explainer_hash=123)
        manager.cache = cache
        with pytest.raises(ValueError, match="Either mask or mask_hash must be provided"):
            _ = manager.extract()

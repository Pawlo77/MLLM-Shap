"""Unit tests for ExplainerCache class (new implementation)."""

from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import torch
from mllm_shap.connectors.base.chat import BaseMllmChat
from mllm_shap.connectors.base.explainer_cache import ExplainerCache
from mllm_shap.connectors.base.model_response import ModelResponse

from ...dummy import DummyChat


class TestExplainerCache:
    """Tests for the new ExplainerCache implementation."""

    @staticmethod
    @pytest.fixture
    def chat() -> BaseMllmChat:
        """Fixture for DummyChat instance."""
        return DummyChat(num_tokens=5)

    @staticmethod
    @pytest.fixture
    def base_responses() -> list[ModelResponse]:
        """Fixture with dummy responses."""
        return [
            ModelResponse(
                chat=None,
                generated_audio_tokens=torch.zeros(1, 2),
                generated_modality_flag=torch.zeros(1, 2),
                generated_text_tokens=torch.zeros(1, 2),
            )
            for _ in range(2)
        ]

    @staticmethod
    @pytest.fixture
    def cache(chat: BaseMllmChat, base_responses: list[ModelResponse]) -> ExplainerCache:
        """Fixture for ExplainerCache instance."""
        masks = torch.ones(2, 3, dtype=torch.bool)
        return ExplainerCache(
            chat=chat,
            calculated_by=111,
            n=3,
            responses=base_responses,
            masks=masks,
            shap_values_mask=chat.shap_values_mask,
        )

    def test_init_extends_masks(self, chat: BaseMllmChat, base_responses: list[ModelResponse]) -> None:
        """Test that masks are extended to match chat length."""
        masks = torch.ones(2, 3, dtype=torch.bool)
        cache = ExplainerCache(
            chat=chat,
            calculated_by=123,
            n=3,
            responses=base_responses,
            masks=masks,
            shap_values_mask=chat.shap_values_mask,
        )
        assert cache.masks.shape == (2, chat.input_tokens_num)
        assert torch.all(cache.masks[:, :3])
        assert torch.all(~cache.masks[:, 3:])  # padded False

    def test_init_raises_if_masks_larger_than_chat(
        self, chat: BaseMllmChat, base_responses: list[ModelResponse]
    ) -> None:
        """Raise if mask has more tokens than chat."""
        masks = torch.ones(2, chat.input_tokens_num + 1, dtype=torch.bool)
        with pytest.raises(ValueError, match="Masks size is larger than the number of tokens"):
            ExplainerCache(
                chat=chat,
                calculated_by=1,
                n=5,
                responses=base_responses,
                masks=masks,
                shap_values_mask=chat.shap_values_mask,
            )

    def test_extend_values_adds_padding(self, chat: BaseMllmChat) -> None:
        """Test extend_values correctly appends fill values."""
        base = torch.tensor([[1.0, 2.0]])
        extended = ExplainerCache.extend_values(
            values=base, shape=(1, 3), dim=1, fill_value=0.0, device=chat.torch_device
        )
        expected = torch.tensor([[1.0, 2.0, 0.0, 0.0, 0.0]])
        assert torch.equal(extended, expected)
        assert extended.device == chat.torch_device

    @patch.object(BaseMllmChat, "shap_values_mask", new_callable=PropertyMock)
    def test_values_setter_valid(self, mock_shap_mask: MagicMock, cache: ExplainerCache) -> None:
        """Test values setter correctly validates and extends."""
        mock_shap_mask.return_value = torch.tensor([True, True, True, False, False])
        values = torch.tensor([1.0, 2.0, 3.0])
        cache.values = values
        out = cache.values
        assert torch.allclose(out[:3], values)
        assert torch.isnan(out[3:]).all()

    def test_values_setter_raises_nan_in_text(self, cache: ExplainerCache) -> None:
        """Raise if NaNs appear in positions where mask=True."""
        cache.shap_values_mask = torch.tensor([True, True, False, False, False])
        values = torch.tensor([float("nan"), 1.0, float("nan")])
        with pytest.raises(ValueError, match="contain NaN values for text tokens"):
            cache.values = values

    def test_values_setter_raises_non_nan_in_non_text(self, cache: ExplainerCache) -> None:
        """Raise if non-NaN values exist where mask=False."""
        cache.shap_values_mask = torch.tensor([True, True, False, False, False])
        values = torch.tensor([1.0, 2.0, 3.0])
        with pytest.raises(
            ValueError,
            match="contain non-NaN values for text tokens they should not explain",
        ):
            cache.values = values

    def test_values_getter_unset(self, cache: ExplainerCache) -> None:
        """Raise if SHAP values not yet computed."""
        with pytest.raises(ValueError, match="have not been computed yet"):
            _ = cache.values

    def test_values_getter_shape_mismatch(self, chat: BaseMllmChat, cache: ExplainerCache) -> None:
        """Raise if SHAP values shape mismatch."""
        cache._values = torch.randn(chat.input_tokens_num - 1)
        with pytest.raises(ValueError, match="size does not match"):
            _ = cache.values

    def test_normalized_values_cycle(self, chat: BaseMllmChat, cache: ExplainerCache) -> None:
        """Test setting/getting normalized values."""
        values = torch.arange(chat.input_tokens_num, dtype=torch.float)
        cache._normalized_values = values
        out = cache.normalized_values
        assert torch.equal(out, values)

    def test_normalized_values_unset(self, cache: ExplainerCache) -> None:
        """Raise if normalized values not computed."""
        with pytest.raises(ValueError, match="have not been computed yet"):
            _ = cache.normalized_values

    def test_create_classmethod(self, chat: BaseMllmChat, base_responses: list[ModelResponse]) -> None:
        """Test ExplainerCache.create sets fields correctly."""
        masks = torch.ones(2, 3, dtype=torch.bool)
        values = torch.tensor([1.0, 2.0, 3.0, float("nan"), float("nan")])
        normalized = values.clone()
        cache = ExplainerCache.create(
            chat=chat,
            explainer_hash=999,
            responses=base_responses,
            masks=masks,
            values=values[:3],
            normalized_values=normalized[:3],
            shap_values_mask=chat.shap_values_mask,
        )
        assert cache.calculated_by == 999
        assert torch.allclose(cache.values, values, equal_nan=True)
        assert torch.allclose(cache.normalized_values, normalized, equal_nan=True)

    def test_del_resets_references(self, cache: ExplainerCache) -> None:
        """Test that __del__ sets all internal refs to None."""
        cache.__del__()
        assert cache.chat is None
        assert cache.responses is None
        assert cache.masks is None
        assert cache._values is None
        assert cache._normalized_values is None

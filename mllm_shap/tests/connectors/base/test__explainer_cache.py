"""Tests for ExplainerCache class."""

import pytest
import torch
from unittest.mock import patch, PropertyMock, MagicMock
from mllm_shap.connectors.base.chat import BaseMllmChat
from mllm_shap.connectors.base.explainer_cache import ExplainerCache
from ...dummy import DummyChat


class TestExplainerCache:
    """Tests for ExplainerCache methods and validation."""

    @staticmethod
    @pytest.fixture
    def chat() -> DummyChat:
        """Fixture for DummyChat instance."""
        return DummyChat()

    @staticmethod
    @pytest.fixture
    def cache(chat: DummyChat) -> ExplainerCache:
        """Fixture for ExplainerCache instance."""
        return ExplainerCache(chat=chat, calculated_by=123)

    def test_extend_values_adds_padding(self, cache: ExplainerCache, chat: DummyChat) -> None:
        """Test extend_values correctly appends fill values."""
        base = torch.tensor([[1.0, 2.0]])
        extended = cache.extend_values(base, shape=(1, 3), dim=1, fill_value=0.0)
        expected = torch.tensor([[1.0, 2.0, 0.0, 0.0, 0.0]])
        assert torch.equal(extended, expected)
        assert extended.device == chat.torch_device

    @patch.object(BaseMllmChat, "input_tokens_num", new_callable=PropertyMock)
    def test_masks_setter_and_getter(self, mock_input_tokens_num: MagicMock, cache: ExplainerCache) -> None:
        """Test masks setter extends and validates correctly."""
        mock_input_tokens_num.return_value = 10
        cache.reduced_embeddings = torch.randn(3, 128)
        values = torch.tensor([[True, False, True], [True, False, True], [True, False, True]])
        cache.masks = values
        out = cache.masks
        assert out.shape == (3, 10)
        assert torch.equal(out[:, :3], values)

    def test_masks_setter_raises_if_no_embeddings(self, cache: ExplainerCache) -> None:
        """Should raise if reduced_embeddings is missing."""
        values = torch.tensor([[True, False, True]])
        with pytest.raises(ValueError, match="Masks size does not match"):
            cache.masks = values

    @patch.object(BaseMllmChat, "input_tokens_num", new_callable=PropertyMock)
    def test_masks_setter_raises_if_size_mismatch(
        self, mock_input_tokens_num: MagicMock, cache: ExplainerCache
    ) -> None:
        """Should raise if reduced_embeddings and mask lengths differ."""
        mock_input_tokens_num.return_value = 5
        cache.reduced_embeddings = torch.randn(3, 3)
        values = torch.tensor([[True, False]])
        with pytest.raises(ValueError, match="Masks size does not match"):
            cache.masks = values

    @patch.object(BaseMllmChat, "input_tokens_num", new_callable=PropertyMock)
    def test_values_getter_and_setter_valid(self, mock_input_tokens_num: MagicMock, cache: ExplainerCache) -> None:
        """Test values setter and getter with valid data."""
        mock_input_tokens_num.return_value = 3

        cache.n = 2
        valid_values = torch.tensor([1.0, 2.0])
        cache.values = valid_values

        out = cache.values
        assert out.shape == torch.Size([3])
        assert torch.equal(out[:2], valid_values)
        assert torch.isnan(out[2:]).all()

    def test_values_getter_raises_if_unset(self, cache: ExplainerCache) -> None:
        """Getter should raise if values not computed."""
        with pytest.raises(ValueError, match="SHAP values have not been computed yet"):
            _ = cache.values

    def test_values_getter_shape_mismatch(self, cache: ExplainerCache) -> None:
        """Getter should raise if tensor shape mismatches chat length."""
        cache._values = torch.randn(7)
        with pytest.raises(ValueError, match="SHAP values size does not match"):
            _ = cache.values

    def test_values_setter_invalid_shape(self, cache: ExplainerCache) -> None:
        """Setter should raise if shape does not match chat input tokens."""
        cache.n = 3
        wrong_shape = torch.randn(10)
        with pytest.raises(ValueError, match="Values size is larger than the number"):
            cache.values = wrong_shape

    def test_values_setter_nan_in_text_tokens(self, cache: ExplainerCache, chat: DummyChat) -> None:
        """Raise if NaNs appear where text tokens exist."""
        cache.n = 3
        values = torch.full((chat.input_tokens_num,), float("nan"))
        with pytest.raises(ValueError, match="contain NaN values for text tokens"):
            cache.values = values

    @patch.object(BaseMllmChat, "shap_values_mask", new_callable=PropertyMock)
    def test_values_setter_non_nan_in_non_text_tokens(
        self, mock_shap_values_mask: MagicMock, cache: ExplainerCache
    ) -> None:
        """Raise if non-NaN values exist where mask=False."""
        mock_shap_values_mask.return_value = torch.tensor([True, True, False])
        cache.n = 3
        values = torch.tensor([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="contain non-NaN values for text tokens they should not explain"):
            cache.values = values

    def test_normalized_values_setter_and_getter(self, cache: ExplainerCache, chat: DummyChat) -> None:
        """Test normalized SHAP values set/get cycle."""
        cache.n = 3
        valid = torch.arange(chat.input_tokens_num, dtype=torch.float)
        cache._normalized_values = valid
        assert torch.equal(cache.normalized_values, valid)

    def test_normalized_values_getter_unset(self, cache: ExplainerCache) -> None:
        """Raise if normalized values not computed yet."""
        with pytest.raises(ValueError, match="Normalized SHAP values have not been computed yet"):
            _ = cache.normalized_values

    def test_normalized_values_shape_mismatch(self, cache: ExplainerCache) -> None:
        """Raise if normalized values shape does not match."""
        cache._normalized_values = torch.randn(2)
        with pytest.raises(ValueError, match="size does not match"):
            _ = cache.normalized_values

    def test_del_resets_all_references(self, cache: ExplainerCache) -> None:
        """Test __del__ correctly nullifies internal attributes."""
        cache.reduced_embeddings = torch.randn(2, 3)
        cache._values = torch.randn(5)
        cache._normalized_values = torch.randn(5)
        cache._masks = torch.ones(5, dtype=torch.bool)
        cache.__del__()
        assert cache.chat is None
        assert cache.reduced_embeddings is None
        assert cache._values is None
        assert cache._normalized_values is None
        assert cache._masks is None

"""Tests for embedding reducers and external embedding classes."""

import pytest
import torch
from mllm_shap.shap.embeddings import (
    FirstReducer,
    MaxReducer,
    MeanReducer,
    MinReducer,
    SumReducer,
    ZeroReducer,
)


class TestEmbeddingReducers:
    """Tests for embedding reducer classes."""

    @staticmethod
    @pytest.fixture
    def embeddings() -> list[torch.Tensor]:
        """Fixture providing example embeddings of shape (d, k)."""
        return [
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),  # (2, 3)
            torch.tensor([[2.0, 0.0, 1.0], [3.0, 3.0, 3.0]]),  # (2, 3)
        ]

    def test_zero_reducer_returns_unchanged(self, embeddings: list[torch.Tensor]) -> None:
        """Test that ZeroReducer returns stacked embeddings unchanged."""
        reducer = ZeroReducer()
        result = reducer(embeddings)
        expected = torch.stack(embeddings, dim=0)
        torch.testing.assert_close(result, expected)

    def test_zero_reducer_raises_on_size_mismatch(self) -> None:
        """Test that ZeroReducer raises an error if embeddings have mismatched sizes."""
        reducer = ZeroReducer()
        bad_embeddings = [torch.randn(2, 4), torch.randn(3, 4)]
        with pytest.raises(ValueError, match="All embeddings must have the same shape"):
            reducer(bad_embeddings)

    def test_mean_reducer_computes_correct_mean(self, embeddings: list[torch.Tensor]) -> None:
        """Test that MeanReducer computes mean across the first dimension (dim=0)."""
        reducer = MeanReducer()
        expected = torch.stack([emb.mean(dim=0) for emb in embeddings], dim=0)
        result = reducer(embeddings)
        torch.testing.assert_close(result, expected)

    def test_max_reducer_computes_correct_values(self, embeddings: list[torch.Tensor]) -> None:
        """Test that MaxReducer computes the elementwise max along the first dimension (dim=0)."""
        reducer = MaxReducer()
        expected = torch.stack([emb.max(dim=0).values for emb in embeddings], dim=0)
        result = reducer(embeddings)
        torch.testing.assert_close(result, expected)

    def test_min_reducer_computes_correct_values(self, embeddings: list[torch.Tensor]) -> None:
        """Test that MinReducer computes the elementwise min along the first dimension (dim=0)."""
        reducer = MinReducer()
        expected = torch.stack([emb.min(dim=0).values for emb in embeddings], dim=0)
        result = reducer(embeddings)
        torch.testing.assert_close(result, expected)

    def test_sum_reducer_computes_correct_values(self, embeddings: list[torch.Tensor]) -> None:
        """Test that SumReducer computes the elementwise sum along the first dimension (dim=0)."""
        reducer = SumReducer()
        expected = torch.stack([emb.sum(dim=0) for emb in embeddings], dim=0)
        result = reducer(embeddings)
        torch.testing.assert_close(result, expected)

    def test_first_reducer_selects_first_column(self, embeddings: list[torch.Tensor]) -> None:
        """Test that FirstReducer selects the first row (dim index 0)."""
        reducer = FirstReducer()
        expected = torch.stack([emb[0] for emb in embeddings], dim=0)
        result = reducer(embeddings)
        torch.testing.assert_close(result, expected)


class TestExternalEmbeddings:
    """Tests for external embedding classes."""

    # TODO

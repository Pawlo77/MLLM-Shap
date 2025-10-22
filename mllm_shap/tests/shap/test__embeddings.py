"""Tests for embedding reducers and external embedding classes."""

import pytest
import torch
from mllm_shap.shap.embeddings import (
    ZeroReducer,
    MeanReducer,
    MaxReducer,
    MinReducer,
    SumReducer,
    FirstReducer,
    OpenAiEmbedding,
)


class TestEmbeddingReducers:
    """Tests for embedding reducer classes."""

    @staticmethod
    @pytest.fixture
    def embeddings() -> torch.Tensor:
        """Fixture for sample embeddings tensor."""
        return torch.tensor([[1.0, 2.0], [3.0, 0.0], [0.0, 4.0]])

    def test_zero_reducer(self, embeddings: torch.Tensor) -> None:
        """Test ZeroReducer returns the original embeddings."""
        reducer = ZeroReducer()
        out = reducer(embeddings)
        assert torch.equal(out, embeddings)

    def test_mean_reducer(self, embeddings: torch.Tensor) -> None:
        """Test MeanReducer computes the mean correctly."""
        reducer = MeanReducer()
        out = reducer(embeddings)
        expected = embeddings.mean(dim=0)
        assert torch.allclose(out, expected)

    def test_max_reducer(self, embeddings: torch.Tensor) -> None:
        """Test MaxReducer computes the max correctly."""
        reducer = MaxReducer()
        out = reducer(embeddings)
        expected = embeddings.max(dim=0).values
        assert torch.allclose(out, expected)

    def test_min_reducer(self, embeddings: torch.Tensor) -> None:
        """Test MinReducer computes the min correctly."""
        reducer = MinReducer()
        out = reducer(embeddings)
        expected = embeddings.min(dim=0).values
        assert torch.allclose(out, expected)

    def test_sum_reducer(self, embeddings: torch.Tensor) -> None:
        """Test SumReducer computes the sum correctly."""
        reducer = SumReducer()
        out = reducer(embeddings)
        expected = embeddings.sum(dim=0)
        assert torch.allclose(out, expected)

    def test_first_reducer(self, embeddings: torch.Tensor) -> None:
        """Test FirstReducer returns the first embedding."""
        reducer = FirstReducer()
        out = reducer(embeddings)
        expected = embeddings[0]
        assert torch.allclose(out, expected)


class TestExternalEmbeddings:
    """Tests for external embedding classes."""

    def test_openai_embedding_not_implemented(self) -> None:
        """Test that OpenAiEmbedding raises NotImplementedError when called."""
        embedding = OpenAiEmbedding()
        with pytest.raises(NotImplementedError):
            embedding(chat=None)  # chat is required but we can pass None for test

"""Unit tests for BaseEmbeddingReducer class."""

import pytest
import torch
from mllm_shap.shap.base.embeddings import BaseEmbeddingReducer
from torch import Tensor


class DummyReducer(BaseEmbeddingReducer):
    """Concrete subclass for testing BaseEmbeddingReducer."""

    def __call__(self, embeddings: list[Tensor]) -> Tensor:  # type: ignore[override]
        return self._prepare(embeddings)


class TestBaseEmbeddingReducer:
    """Unit tests for BaseEmbeddingReducer initialization and behavior."""

    def test_init_with_valid_n(self) -> None:
        """Should correctly initialize when n is a positive integer."""
        reducer = DummyReducer(n=5)
        assert reducer.n == 5

    def test_init_with_none_n(self) -> None:
        """Should correctly initialize when n is None."""
        reducer = DummyReducer(n=None)
        assert reducer.n is None

    @pytest.mark.parametrize("invalid_n", [0, -3, -1])
    def test_init_raises_for_invalid_n(self, invalid_n: int) -> None:
        """Should raise ValueError if n <= 0."""
        with pytest.raises(ValueError, match="n must be None or a positive integer"):
            DummyReducer(n=invalid_n)

    def test_call_raises_if_embedding_not_tensor(self) -> None:
        """Should raise ValueError if any embedding is not a torch.Tensor."""
        reducer = DummyReducer()
        embeddings = [torch.randn(2, 3), "not_a_tensor"]  # type: ignore[list-item]
        with pytest.raises(ValueError, match="Embedding at index 1 is not a Tensor"):
            _ = reducer(embeddings)

    def test_call_passes_through_when_n_is_none(self) -> None:
        """Should return unchanged embeddings when n=None."""
        reducer = DummyReducer(n=None)
        embeddings = [torch.randn(4, 8) for _ in range(2)]
        result = reducer(embeddings)
        assert all(torch.equal(a, b) for a, b in zip(result, embeddings))

    def test_call_reduces_embeddings_when_n_smaller_than_k(self) -> None:
        """Should randomly sample embeddings when n < k."""
        n = 2
        reducer = DummyReducer(n=n)
        embeddings = [torch.randn(8, 10) for _ in range(3)]
        result = reducer(embeddings)
        assert isinstance(result, list)
        assert all(isinstance(e, Tensor) for e in result)
        # result tensors should have at most n columns in last dimension
        for emb in result:
            assert emb.shape[1] == embeddings[0].shape[1] or emb.shape[-1] <= n

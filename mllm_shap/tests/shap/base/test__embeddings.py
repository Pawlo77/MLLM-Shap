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

    def test_call_returns_same_list_instance(self) -> None:
        """Should mutate the provided list in place rather than creating a new one."""
        reducer = DummyReducer(n=None)
        embeddings = [torch.randn(4, 6) for _ in range(2)]
        result = reducer(embeddings)
        assert result is embeddings

    def test_call_sampling_replaces_tensor_reference(self) -> None:
        """Should replace individual tensors when sampling is applied."""
        reducer = DummyReducer(n=1)
        embeddings = [torch.arange(12, dtype=torch.float32).reshape(3, 4)]
        original = embeddings[0]
        torch.manual_seed(0)
        result = reducer(embeddings)
        assert result is embeddings
        assert result[0] is not original
        assert result[0].shape[-1] == 1

    def test_call_does_not_sample_when_limit_exceeds_dimension(self) -> None:
        """Should not alter tensor references when sampling limit exceeds last dimension."""
        reducer = DummyReducer(n=10)
        embeddings = [torch.arange(12, dtype=torch.float32).reshape(3, 4)]
        original = embeddings[0]
        result = reducer(embeddings)
        assert result[0] is original

    def test_call_handles_empty_embeddings_list(self) -> None:
        """Should safely handle empty inputs."""
        reducer = DummyReducer(n=2)
        embeddings: list[Tensor] = []
        result = reducer(embeddings)
        assert result is embeddings
        assert result == []

    def test_call_handles_zero_length_tensor(self) -> None:
        """Should return tensors with zero samples without raising an error."""
        reducer = DummyReducer(n=2)
        embeddings = [torch.empty((0, 5))]
        torch.manual_seed(0)
        result = reducer(embeddings)
        assert result[0].shape[0] == 0
        assert result[0].shape[-1] == 0

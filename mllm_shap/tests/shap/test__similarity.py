"""Tests for similarity metrics used in SHAP explainers."""

import pytest
import torch
from mllm_shap.shap.similarity import CosineSimilarity


@pytest.fixture
def base_embedding() -> torch.Tensor:
    """Fixture for base embedding tensor."""
    return torch.tensor([1.0, 0.0])


@pytest.fixture
def other_embeddings() -> torch.Tensor:
    """Fixture for other embeddings tensor."""
    return torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )


class TestCosineSimilarity:
    """Tests for CosineSimilarity class."""

    def test_identical_vectors(self, base_embedding: torch.Tensor, other_embeddings: torch.Tensor) -> None:
        """Test similarity of identical and orthogonal vectors."""
        similarity = CosineSimilarity()
        # first vector is identical to base
        out = similarity(base_embedding, other_embeddings)
        assert torch.isclose(out[0], torch.tensor(1.0))
        # second vector is orthogonal
        assert torch.isclose(out[1], torch.tensor(0.0))
        # third vector at 45° angle
        expected = 1 / (2**0.5)
        assert torch.isclose(out[2], torch.tensor(expected), atol=1e-5)

    def test_zero_vector_base(self) -> None:
        """Test similarity when base vector is zero."""
        similarity = CosineSimilarity()
        base_emb = torch.tensor([0.0, 0.0])
        other_embs = torch.tensor([[1.0, 0.0]])
        # Should not throw, but returns nan due to zero division
        out = similarity(base_emb, other_embs)
        assert torch.isnan(out[0])

    def test_zero_vector_others(self) -> None:
        """Test similarity when other vectors contain zero vector."""
        similarity = CosineSimilarity()
        base_emb = torch.tensor([1.0, 0.0])
        other_embs = torch.tensor([[0.0, 0.0]])
        out = similarity(base_emb, other_embs)
        assert torch.isnan(out[0])

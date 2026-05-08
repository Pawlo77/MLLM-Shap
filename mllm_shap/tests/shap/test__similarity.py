"""Tests for similarity metrics used in SHAP explainers."""

import pytest
import torch
from mllm_shap.shap.similarity import (
    CosineSimilarity,
    TfIdfCosineSimilarity,
    EuclideanSimilarity,
)
from mllm_shap.connectors.base.model_response import ModelResponse


class TestEuclideanSimilarity:
    """Tests for EuclideanSimilarity class."""

    def test_basic_similarity(self):
        """Checks that basic similarity."""
        similarity = EuclideanSimilarity()
        base = torch.tensor([0.0, 0.0])
        others = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
        out = similarity(base, others)
        expected = torch.tensor([1.0, 1 / (1 + 1.0), 1 / (1 + 2.0)])
        assert torch.allclose(out, expected, atol=1e-5)

    def test_nonzero_base(self):
        """Checks that nonzero base."""
        similarity = EuclideanSimilarity()
        base = torch.tensor([1.0, 1.0])
        others = torch.tensor([[1.0, 1.0], [2.0, 1.0]])
        out = similarity(base, others)
        expected = torch.tensor([1.0, 1 / (1 + 1.0)])
        assert torch.allclose(out, expected, atol=1e-5)

    def test_similarity_monotonic_with_distance(self):
        """Checks that similarity monotonic with distance."""
        similarity = EuclideanSimilarity()
        base = torch.zeros(3)
        others = torch.tensor([[0.1, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        out = similarity(base, others)
        assert out[0] > out[1] > out[2]

    def test_preserves_dtype(self):
        """Checks that preserves dtype."""
        similarity = EuclideanSimilarity()
        base = torch.tensor([0.0, 1.0], dtype=torch.float64)
        others = torch.tensor([[0.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
        out = similarity(base, others)
        assert out.dtype == torch.float64


class TestCosineSimilarity:
    """Tests for CosineSimilarity class."""

    def test_identical_vectors(self):
        """Checks that identical vectors."""
        similarity = CosineSimilarity()
        base = torch.tensor([1.0, 0.0])
        others = torch.tensor(
            [
                [1.0, 0.0],  # identical
                [0.0, 1.0],  # orthogonal
                [1.0, 1.0],  # 45 degree
            ]
        )
        out = similarity(base, others)
        expected = torch.tensor([1.0, 0.0, 1 / (2**0.5)])
        assert torch.allclose(out, expected, atol=1e-5)

    def test_zero_vector_base(self):
        """Checks that zero vector base."""
        similarity = CosineSimilarity()
        base = torch.tensor([0.0, 0.0])
        others = torch.tensor([[1.0, 0.0]])
        out = similarity(base, others)
        # Zero vectors are clamped for numerical stability, resulting in ~0 similarity
        assert torch.allclose(out[0], torch.tensor(0.0), atol=1e-5)

    def test_zero_vector_others(self):
        """Checks that zero vector others."""
        similarity = CosineSimilarity()
        base = torch.tensor([1.0, 0.0])
        others = torch.tensor([[0.0, 0.0]])
        out = similarity(base, others)
        # Zero vectors are clamped for numerical stability, resulting in ~0 similarity
        assert torch.allclose(out[0], torch.tensor(0.0), atol=1e-5)

    def test_scaling_invariance(self):
        """Checks that scaling invariance."""
        similarity = CosineSimilarity()
        base = torch.tensor([1.0, 2.0])
        others = torch.tensor([[2.0, 4.0], [0.5, 1.0]])
        out = similarity(base, others)
        assert torch.allclose(out, torch.ones_like(out), atol=1e-6)

    def test_values_within_expected_range(self):
        """Checks that values within expected range."""
        similarity = CosineSimilarity()
        base = torch.tensor([1.0, -1.0])
        others = torch.tensor([[1.0, 1.0], [-1.0, 1.0]])
        out = similarity(base, others)
        assert torch.all(out <= 1.0)
        assert torch.all(out >= -1.0)

    def test_output_dtype_matches_input(self):
        """Checks that output dtype matches input."""
        similarity = CosineSimilarity()
        base = torch.tensor([1.0, 0.0], dtype=torch.float64)
        others = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        out = similarity(base, others)
        assert out.dtype == torch.float64


class TestTfIdfCosineSimilarity:
    """Tests for TfIdfCosineSimilarity class."""

    def make_model_response(self, text_tokens, audio_tokens):
        return ModelResponse(
            chat=None,
            generated_text_tokens=torch.tensor(text_tokens, dtype=torch.float32),
            generated_audio_tokens=torch.tensor(audio_tokens, dtype=torch.float32),
            generated_modality_flag=torch.zeros(1),  # dummy
        )

    def test_basic_similarity(self):
        """Checks that basic similarity."""
        similarity = TfIdfCosineSimilarity()
        base = self.make_model_response([[1, 0], [0, 1]], [[0, 1], [1, 0]])
        others = [
            base,  # first must equal base
            self.make_model_response([[1, 0], [0, 1]], [[0, 1], [1, 0]]),
        ]
        out = similarity(base, others)
        assert out.shape[0] == len(others)
        assert torch.allclose(out.clamp(0, 1), out, atol=1e-6)

    def test_similarity_penalizes_token_changes(self):
        """Checks that similarity penalizes token changes."""
        similarity = TfIdfCosineSimilarity()
        base = self.make_model_response([[1, 0]], [[0, 1]])
        altered = self.make_model_response([[1, 0], [0, 1]], [[0, 1]])
        others = [base, altered]
        out = similarity(base, others)
        assert torch.isclose(out[0], torch.tensor(1.0, device=out.device), atol=1e-6)
        assert out[1] < out[0]

    def test_output_device_matches_base(self):
        """Checks that output device matches base."""
        similarity = TfIdfCosineSimilarity()
        base = self.make_model_response([[1, 0]], [[0, 1]])
        others = [base, self.make_model_response([[1, 0]], [[1, 0]])]
        out = similarity(base, others)
        assert out.device == base.generated_text_tokens.device

    def test_operates_on_embeddings_flag(self):
        """Checks that operates on embeddings flag."""
        similarity = TfIdfCosineSimilarity()
        assert similarity.operates_on_embeddings is False

    def test_first_element_must_be_base(self):
        """Checks that first element must be base."""
        similarity = TfIdfCosineSimilarity()
        base = self.make_model_response([[1, 0]], [[0, 1]])
        others = [self.make_model_response([[1, 0]], [[0, 1]])]  # not base object
        # forcibly make first element different
        others[0].generated_text_tokens += 1
        with pytest.raises(ValueError):
            similarity(base, others)

    def test_tokenization_consistency(self):
        """Checks that tokenization consistency."""
        similarity = TfIdfCosineSimilarity()
        resp = self.make_model_response([[1, 2], [3, 4]], [[5, 6]])
        token1 = similarity._TfIdfCosineSimilarity__tokenize(resp.generated_text_tokens)
        token2 = similarity._TfIdfCosineSimilarity__tokenize(resp.generated_text_tokens)
        assert torch.equal(token1, token2)  # tokenization is consistent

    def test_tokenization_produces_unique_ids(self):
        """Checks that tokenization produces unique ids."""
        similarity = TfIdfCosineSimilarity()
        tokens1 = similarity._TfIdfCosineSimilarity__tokenize(
            torch.tensor([[1, 2], [3, 4]])
        )
        tokens2 = similarity._TfIdfCosineSimilarity__tokenize(
            torch.tensor([[2, 3], [4, 5]])
        )
        assert not torch.equal(tokens1, tokens2)

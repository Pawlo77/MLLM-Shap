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
        similarity = EuclideanSimilarity()
        base = torch.tensor([0.0, 0.0])
        others = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
        out = similarity(base, others)
        expected = torch.tensor([1.0, 1 / (1 + 1.0), 1 / (1 + 2.0)])
        assert torch.allclose(out, expected, atol=1e-5)

    def test_nonzero_base(self):
        similarity = EuclideanSimilarity()
        base = torch.tensor([1.0, 1.0])
        others = torch.tensor([[1.0, 1.0], [2.0, 1.0]])
        out = similarity(base, others)
        expected = torch.tensor([1.0, 1 / (1 + 1.0)])
        assert torch.allclose(out, expected, atol=1e-5)


class TestCosineSimilarity:
    """Tests for CosineSimilarity class."""

    def test_identical_vectors(self):
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
        similarity = CosineSimilarity()
        base = torch.tensor([0.0, 0.0])
        others = torch.tensor([[1.0, 0.0]])
        out = similarity(base, others)
        assert torch.isnan(out[0])

    def test_zero_vector_others(self):
        similarity = CosineSimilarity()
        base = torch.tensor([1.0, 0.0])
        others = torch.tensor([[0.0, 0.0]])
        out = similarity(base, others)
        assert torch.isnan(out[0])


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
        similarity = TfIdfCosineSimilarity()
        base = self.make_model_response([[1, 0], [0, 1]], [[0, 1], [1, 0]])
        others = [
            base,  # first must equal base
            self.make_model_response([[1, 0], [0, 1]], [[0, 1], [1, 0]]),
        ]
        out = similarity(base, others)
        assert out.shape[0] == len(others)
        assert torch.allclose(out.clamp(0, 1), out, atol=1e-6)

    def test_first_element_must_be_base(self):
        similarity = TfIdfCosineSimilarity()
        base = self.make_model_response([[1, 0]], [[0, 1]])
        others = [self.make_model_response([[1, 0]], [[0, 1]])]  # not base object
        # forcibly make first element different
        others[0].generated_text_tokens += 1
        with pytest.raises(ValueError):
            similarity(base, others)

    def test_tokenization_consistency(self):
        similarity = TfIdfCosineSimilarity()
        resp = self.make_model_response([[1, 2], [3, 4]], [[5, 6]])
        token1 = similarity._TfIdfCosineSimilarity__tokenize(resp.generated_text_tokens)
        token2 = similarity._TfIdfCosineSimilarity__tokenize(resp.generated_text_tokens)
        assert torch.equal(token1, token2)  # tokenization is consistent

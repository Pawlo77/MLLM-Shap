"""Tests for embedding reducers and external embedding classes."""

import pytest
import torch
from mllm_shap.connectors.base.model_response import ModelResponse
from mllm_shap.shap.embeddings import (
    CustomEmbedding,
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

    def test_zero_reducer_returns_unchanged(
        self, embeddings: list[torch.Tensor]
    ) -> None:
        """Test that ZeroReducer returns stacked embeddings unchanged."""
        reducer = ZeroReducer()
        result = reducer(embeddings)
        expected = torch.stack(embeddings, dim=0)
        torch.testing.assert_close(result, expected)
        assert result is not expected

    def test_zero_reducer_raises_on_size_mismatch(self) -> None:
        """Test that ZeroReducer raises an error if embeddings have mismatched sizes."""
        reducer = ZeroReducer()
        bad_embeddings = [torch.randn(2, 4), torch.randn(3, 4)]
        with pytest.raises(ValueError, match="All embeddings must have the same shape"):
            reducer(bad_embeddings)

    def test_zero_reducer_rejects_non_tensor_inputs(self) -> None:
        """ZeroReducer should raise when encountering non-Tensor inputs."""
        reducer = ZeroReducer()
        with pytest.raises(ValueError, match="Embedding at index 0 is not a Tensor"):
            reducer(["not-a-tensor"])

    def test_zero_reducer_respects_sampling_limit(
        self, embeddings: list[torch.Tensor]
    ) -> None:
        """ZeroReducer should cap the number of vectors according to the sampling limit."""
        torch.manual_seed(0)
        reducer = ZeroReducer(n=1)
        result = reducer([emb.clone() for emb in embeddings])
        assert result.shape[-1] == 1

    def test_mean_reducer_computes_correct_mean(
        self, embeddings: list[torch.Tensor]
    ) -> None:
        """Test that MeanReducer computes mean across the first dimension (dim=0)."""
        reducer = MeanReducer()
        expected = torch.stack([emb.mean(dim=0) for emb in embeddings], dim=0)
        result = reducer(embeddings)
        torch.testing.assert_close(result, expected)

    def test_mean_reducer_respects_sampling_limit(
        self, embeddings: list[torch.Tensor]
    ) -> None:
        """MeanReducer should apply the optional sampling limit before reduction."""
        torch.manual_seed(0)
        reducer = MeanReducer(n=1)
        inputs = [emb.clone() for emb in embeddings]
        expected = []
        for emb in inputs:
            indices = torch.randperm(emb.shape[0])[:1]
            sampled = emb[..., indices]
            expected.append(sampled.mean(dim=0))
        expected_tensor = torch.stack(expected, dim=0)
        torch.manual_seed(0)
        result = reducer([emb.clone() for emb in embeddings])
        torch.testing.assert_close(result, expected_tensor)

    def test_max_reducer_computes_correct_values(
        self, embeddings: list[torch.Tensor]
    ) -> None:
        """Test that MaxReducer computes the elementwise max along the first dimension (dim=0)."""
        reducer = MaxReducer()
        expected = torch.stack([emb.max(dim=0).values for emb in embeddings], dim=0)
        result = reducer(embeddings)
        torch.testing.assert_close(result, expected)

    def test_max_reducer_preserves_dtype(self, embeddings: list[torch.Tensor]) -> None:
        """MaxReducer output dtype should match input dtype."""
        reducer = MaxReducer()
        typed_embeddings = [emb.double() for emb in embeddings]
        result = reducer(typed_embeddings)
        assert result.dtype == torch.float64

    def test_min_reducer_computes_correct_values(
        self, embeddings: list[torch.Tensor]
    ) -> None:
        """Test that MinReducer computes the elementwise min along the first dimension (dim=0)."""
        reducer = MinReducer()
        expected = torch.stack([emb.min(dim=0).values for emb in embeddings], dim=0)
        result = reducer(embeddings)
        torch.testing.assert_close(result, expected)

    def test_min_reducer_with_single_vector(self) -> None:
        """Reducer should handle inputs with a single sample without error."""
        reducer = MinReducer()
        embeddings = [torch.tensor([[2.0, 3.0, 4.0]])]
        result = reducer(embeddings)
        torch.testing.assert_close(result, torch.tensor([[2.0, 3.0, 4.0]]))

    def test_sum_reducer_computes_correct_values(
        self, embeddings: list[torch.Tensor]
    ) -> None:
        """Test that SumReducer computes the elementwise sum along the first dimension (dim=0)."""
        reducer = SumReducer()
        expected = torch.stack([emb.sum(dim=0) for emb in embeddings], dim=0)
        result = reducer(embeddings)
        torch.testing.assert_close(result, expected)

    def test_sum_reducer_sampling_limit(self, embeddings: list[torch.Tensor]) -> None:
        """SumReducer should honor the optional sampling parameter."""
        torch.manual_seed(1)
        reducer = SumReducer(n=1)
        inputs = [emb.clone() for emb in embeddings]
        expected = []
        for emb in inputs:
            indices = torch.randperm(emb.shape[0])[:1]
            sampled = emb[..., indices]
            expected.append(sampled.sum(dim=0))
        expected_tensor = torch.stack(expected, dim=0)
        torch.manual_seed(1)
        result = reducer([emb.clone() for emb in embeddings])
        torch.testing.assert_close(result, expected_tensor)

    def test_first_reducer_selects_first_column(
        self, embeddings: list[torch.Tensor]
    ) -> None:
        """Test that FirstReducer selects the first row (dim index 0)."""
        reducer = FirstReducer()
        expected = torch.stack([emb[0] for emb in embeddings], dim=0)
        result = reducer(embeddings)
        torch.testing.assert_close(result, expected)

    def test_reducer_init_rejects_invalid_n(self) -> None:
        """All reducers should reject invalid sampler limits."""
        with pytest.raises(ValueError, match="n must be None or a positive integer"):
            MeanReducer(n=0)


class TestExternalEmbeddings:
    """Tests for external embedding classes."""

    class _GenerationTokenizerStub:
        """Tokenizer stub used to decode generated token ids."""

        def decode(self, token_ids: list[int], skip_special_tokens: bool = False):
            del skip_special_tokens
            return f"tok{token_ids[0]}"

    class _EmbeddingTokenizerStub:
        """Embedding tokenizer stub returning padded tensors."""

        def __call__(
            self,
            batch: list[str],
            padding: bool,
            truncation: bool,
            max_length: int,
            return_tensors: str,
        ):
            del padding, truncation, max_length, return_tensors
            b = len(batch)
            ids = torch.arange(1, b + 1, dtype=torch.long).unsqueeze(1)
            attn = torch.ones((b, 1), dtype=torch.long)
            return {"input_ids": ids, "attention_mask": attn}

    class _EmbeddingModelStub:
        """Embedding model stub exposing HF-like API."""

        def __init__(self) -> None:
            self.config = type("Cfg", (), {"hidden_size": 4})()

        def to(
            self, device: torch.device
        ) -> "TestExternalEmbeddings._EmbeddingModelStub":
            del device
            return self

        def eval(self) -> "TestExternalEmbeddings._EmbeddingModelStub":
            return self

        def __call__(self, **inputs):
            input_ids = inputs["input_ids"].to(dtype=torch.float32)  # [B, 1]
            # last_hidden_state: [B, 1, H]
            last_hidden = input_ids.unsqueeze(-1).repeat(1, 1, 4)
            return type("Out", (), {"last_hidden_state": last_hidden})()

    def test_custom_embedding_rejects_non_sha_revision(self) -> None:
        """CustomEmbedding should validate commit hash revision."""
        with pytest.raises(ValueError, match="40-character hex commit SHA"):
            _ = CustomEmbedding(
                generation_tokenizer=self._GenerationTokenizerStub(),
                embed_model_id="stub/model",
                embed_revision="notasha",
                device=torch.device("cpu"),
            )

    def test_custom_embedding_handles_empty_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty generated tokens should return empty embedding tensor."""
        monkeypatch.setattr(
            "mllm_shap.shap.embeddings.AutoTokenizer.from_pretrained",
            lambda *args, **kwargs: self._EmbeddingTokenizerStub(),
        )
        monkeypatch.setattr(
            "mllm_shap.shap.embeddings.AutoModel.from_pretrained",
            lambda *args, **kwargs: self._EmbeddingModelStub(),
        )
        emb = CustomEmbedding(
            generation_tokenizer=self._GenerationTokenizerStub(),
            embed_model_id="stub/model",
            embed_revision="a" * 40,
            device=torch.device("cpu"),
        )
        out = emb(
            [
                ModelResponse(
                    chat=None,
                    generated_text_tokens=torch.tensor([], dtype=torch.long),
                    generated_audio_tokens=torch.empty((0, 0), dtype=torch.long),
                    generated_modality_flag=torch.empty((0,), dtype=torch.long),
                )
            ]
        )
        assert len(out) == 1
        assert out[0].shape == (0, 4)

    def test_custom_embedding_outputs_per_token_vectors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CustomEmbedding should return one vector per generated token."""
        monkeypatch.setattr(
            "mllm_shap.shap.embeddings.AutoTokenizer.from_pretrained",
            lambda *args, **kwargs: self._EmbeddingTokenizerStub(),
        )
        monkeypatch.setattr(
            "mllm_shap.shap.embeddings.AutoModel.from_pretrained",
            lambda *args, **kwargs: self._EmbeddingModelStub(),
        )
        emb = CustomEmbedding(
            generation_tokenizer=self._GenerationTokenizerStub(),
            embed_model_id="stub/model",
            embed_revision="b" * 40,
            device=torch.device("cpu"),
            l2_normalize=False,
        )
        responses = [
            ModelResponse(
                chat=None,
                generated_text_tokens=torch.tensor([5, 6, 7], dtype=torch.long),
                generated_audio_tokens=torch.empty((0, 0), dtype=torch.long),
                generated_modality_flag=torch.tensor([0, 0, 0], dtype=torch.long),
            )
        ]
        out = emb(responses)
        assert len(out) == 1
        assert out[0].shape == (3, 4)

# pylint: disable=too-few-public-methods

"""Embedding calculation and reduction strategies for SHAP explanations."""

import re
from typing import List, Sequence, cast

from torch import Tensor
import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedModel
)

from ..connectors.base.model_response import ModelResponse
from .base.embeddings import BaseEmbeddingReducer, BaseExternalEmbedding


class ZeroReducer(BaseEmbeddingReducer):
    """Dummy reducer that returns embeddings unchanged."""

    def __call__(self, embeddings: list[Tensor]) -> Tensor:
        embeddings = self._prepare(embeddings)

        shapes = [tuple(e.shape) for e in embeddings]
        if len(set(shapes)) != 1:
            raise ValueError(f"All embeddings must have the same shape for ZeroReducer. " f"Got shapes: {shapes}")

        return torch.stack(embeddings, dim=0)


class MeanReducer(BaseEmbeddingReducer):
    """Reducer that computes the mean of embeddings."""

    def __call__(self, embeddings: list[Tensor]) -> Tensor:
        embeddings = self._prepare(embeddings)

        for i, emb in enumerate(embeddings):
            embeddings[i] = emb.mean(dim=0)
        return torch.stack(embeddings, dim=0)


class MaxReducer(BaseEmbeddingReducer):
    """Reducer that computes the max of embeddings."""

    def __call__(self, embeddings: list[Tensor]) -> Tensor:
        embeddings = self._prepare(embeddings)

        for i, emb in enumerate(embeddings):
            embeddings[i] = emb.max(dim=0).values
        return torch.stack(embeddings, dim=0)


class MinReducer(BaseEmbeddingReducer):
    """Reducer that computes the min of embeddings."""

    def __call__(self, embeddings: list[Tensor]) -> Tensor:
        embeddings = self._prepare(embeddings)

        for i, emb in enumerate(embeddings):
            embeddings[i] = emb.min(dim=0).values
        return torch.stack(embeddings, dim=0)


class SumReducer(BaseEmbeddingReducer):
    """Reducer that computes the sum of embeddings."""

    def __call__(self, embeddings: list[Tensor]) -> Tensor:
        embeddings = self._prepare(embeddings)

        for i, emb in enumerate(embeddings):
            embeddings[i] = emb.sum(dim=0)
        return torch.stack(embeddings, dim=0)


class FirstReducer(BaseEmbeddingReducer):
    """
    Reducer that selects the first embedding.

    :attr:`n` parameter is ignored in this reducer.
    """

    def __call__(self, embeddings: list[Tensor]) -> Tensor:
        embeddings = self._prepare(embeddings)

        for i, emb in enumerate(embeddings):
            embeddings[i] = emb[..., 0, :]
        return torch.stack(embeddings, dim=0)


class CustomEmbedding(BaseExternalEmbedding):  # pylint: disable=too-many-instance-attributes
    """
    External embeddings using a **local** encoder model (e.g., E5/SBERT).

    For each :class:`ModelResponse`, we:
      1) take ``generated_text_tokens`` (shape [T]),
      2) decode **each token id** with the **generation tokenizer** to a short text piece,
      3) embed each piece independently with the **embedding encoder**,
      4) return a tensor of shape **[T, hidden]** per response (aligned 1:1 with tokens).
    """

    tokenizer_decode: PreTrainedTokenizerBase
    emb_tokenizer: PreTrainedTokenizerBase
    emb_model: PreTrainedModel
    device: torch.device
    max_length: int
    batch_size: int
    l2_normalize: bool
    _hidden_size: int

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        *,
        generation_tokenizer: PreTrainedTokenizerBase,
        embed_model_id: str,
        embed_revision: str,
        device: torch.device,
        max_length: int = 64,
        batch_size: int = 64,
        l2_normalize: bool = True,
        local_files_only: bool = True,
    ) -> None:
        """
        Args:
            generation_tokenizer: Tokenizer used by the **generation LM** (for token-id -> text decoding).
            embed_model_id: Local HF model id/path for the **encoder** (e.g., ``"intfloat/e5-base-v2"``).
            embed_revision: **40-hex commit SHA** for the encoder model (immutable).
            device: Torch device for inference.
            max_length: Max tokens per decoded piece for the embedding tokenizer.
            batch_size: Per-forward batch size for encoder inference.
            l2_normalize: L2-normalize embeddings.
            local_files_only: Enforce local loading (no network); set False if you allow online fetch.
        """
        # Security: enforce immutable commit SHA (satisfies Bandit B615 when loading with revision).
        if not isinstance(embed_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", embed_revision):
            raise ValueError("embed_revision must be a 40-character hex commit SHA.")

        self.tokenizer_decode = generation_tokenizer

        self.emb_tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            embed_model_id,
            revision=embed_revision,  # nosec: B615 - pinned commit
            local_files_only=local_files_only,
        )
        _model = AutoModel.from_pretrained(
            embed_model_id,
            revision=embed_revision,  # nosec: B615 - pinned commit
            local_files_only=local_files_only,
        )
        self.emb_model = cast(PreTrainedModel, _model)
        self.emb_model.to(device)  # type: ignore[arg-type]
        self.emb_model.eval()      # type: ignore[no-untyped-call]

        self.device = device
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.l2_normalize = bool(l2_normalize)
        cfg = getattr(self.emb_model, "config", None)
        hidden_size = getattr(cfg, "hidden_size", None)
        self._hidden_size = int(hidden_size) if isinstance(hidden_size, int) else 768

    def __call__(self, responses: list[ModelResponse]) -> list[Tensor]:
        """
        Compute **per-token** embeddings for each response.

        Returns:
            list[Tensor]: For each response, a tensor of shape **[T, hidden]** (T = tokens in response).
        """
        result: list[Tensor] = []
        for resp in responses:
            token_ids: Tensor = resp.generated_text_tokens  # [T]
            if token_ids.numel() == 0:
                result.append(torch.empty(0, self._hidden_size, dtype=torch.float32, device=self.device))
                continue

            # Decode each token id to its text piece (keep specials to preserve alignment)
            pieces: List[str] = [
                self.tokenizer_decode.decode([int(tid)], skip_special_tokens=False) for tid in token_ids.tolist()
            ]

            emb = self._embed_texts(pieces)  # [T, hidden]
            result.append(emb)
        return result

    @torch.inference_mode()
    def _embed_texts(self, texts: Sequence[str]) -> Tensor:
        """Embed a list of short texts -> [len(texts), hidden]."""
        vecs: list[Tensor] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i: i + self.batch_size])

            inputs = self.emb_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.emb_model(**inputs)
            last_hidden: Tensor = outputs.last_hidden_state  # [B, L, H]
            attn: Tensor = inputs["attention_mask"].unsqueeze(-1).type_as(last_hidden)  # [B, L, 1]

            # Mean pool over non-padding tokens
            summed: Tensor = (last_hidden * attn).sum(dim=1)  # [B, H]
            lengths: Tensor = attn.sum(dim=1).clamp(min=1.0)  # [B, 1]
            mean: Tensor = summed / lengths  # [B, H]

            if self.l2_normalize:
                mean = torch.nn.functional.normalize(mean, p=2, dim=1)

            vecs.append(mean)

        return torch.cat(vecs, dim=0)

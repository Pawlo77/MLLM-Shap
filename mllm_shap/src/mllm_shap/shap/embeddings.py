"""Embedding calculation and reduction strategies for SHAP explanations."""

import re
from typing import Any, List, Sequence, cast

from torch import Tensor
import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)

from ..connectors.base.model_response import ModelResponse
from .base.embeddings import BaseEmbeddingReducer, BaseExternalEmbedding


class ZeroReducer(BaseEmbeddingReducer):
    """Reducer that flattens and stacks embeddings into (N, S*H)."""

    def __call__(self, embeddings: list[Tensor]) -> Tensor:
        embeddings = self._prepare(embeddings)

        shapes = [tuple(e.shape) for e in embeddings]
        if len(set(shapes)) != 1:
            raise ValueError(
                f"All embeddings must have the same shape for ZeroReducer flattening. "
                f"Got shapes: {shapes}"
            )

        # Flatten each tensor from (S, H) to (S*H,)
        # then stack into (N, S*H)
        flattened = [e.reshape(-1) for e in embeddings]
        return torch.stack(flattened, dim=0)


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


class CustomEmbedding(BaseExternalEmbedding):
    """
    External embeddings using a **local** encoder model (e.g., E5/SBERT).

    For each :class:`ModelResponse`, we:
    1) take ``generated_text_tokens`` (shape [T]),
    2) decode **each token id** with the **generation tokenizer** to a short text piece,
    3) embed each piece independently with the **embedding encoder*
    4) return a tensor of shape **[T, hidden]** per response (aligned 1:1 with tokens).
    """

    tokenizer_decode: PreTrainedTokenizerBase
    """Tokenizer from the generation model used to decode token IDs into text pieces."""
    emb_tokenizer: PreTrainedTokenizerBase
    """Tokenizer used by the external embedding encoder model."""
    emb_model: PreTrainedModel
    """External embedding encoder model used for text piece embedding."""
    device: torch.device
    """Torch device where embedding inference is executed."""
    max_length: int
    """Maximum sequence length passed to the embedding tokenizer."""
    batch_size: int
    """Batch size used for encoder forward passes during embedding."""
    l2_normalize: bool
    """Whether to L2-normalize pooled embedding vectors."""
    _hidden_size: int
    """Fallback/derived embedding width used for empty-response tensors."""

    def __init__(
        self,
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
        if not isinstance(embed_revision, str) or not re.fullmatch(
            r"[0-9a-f]{40}", embed_revision
        ):
            raise ValueError("embed_revision must be a 40-character hex commit SHA.")

        self.tokenizer_decode = generation_tokenizer

        self.emb_tokenizer = cast(Any, AutoTokenizer).from_pretrained(
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
        cast(Any, self.emb_model).to(device)
        cast(Any, self.emb_model).eval()

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
                result.append(
                    torch.empty(
                        0, self._hidden_size, dtype=torch.float32, device=self.device
                    )
                )
                continue

            # Batch-decode all token IDs at once for efficiency
            ids_list = [[int(tid)] for tid in token_ids.tolist()]
            if hasattr(self.tokenizer_decode, "batch_decode"):
                pieces: List[str] = self.tokenizer_decode.batch_decode(
                    ids_list, skip_special_tokens=False
                )
            else:
                pieces = [
                    self.tokenizer_decode.decode(ids, skip_special_tokens=False)
                    for ids in ids_list
                ]

            emb = self._embed_texts(pieces)  # [T, hidden]
            result.append(emb)
        return result

    @torch.inference_mode()
    def _embed_texts(self, texts: Sequence[str]) -> Tensor:
        """Embed a list of short texts -> [len(texts), hidden]."""
        vecs: list[Tensor] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])

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
            attn: Tensor = (
                inputs["attention_mask"].unsqueeze(-1).type_as(last_hidden)
            )  # [B, L, 1]

            # Mean pool over non-padding tokens
            summed: Tensor = (last_hidden * attn).sum(dim=1)  # [B, H]
            lengths: Tensor = attn.sum(dim=1).clamp(min=1.0)  # [B, 1]
            mean: Tensor = summed / lengths  # [B, H]

            if self.l2_normalize:
                mean = torch.nn.functional.normalize(mean, p=2, dim=1)

            vecs.append(mean)

        return torch.cat(vecs, dim=0)

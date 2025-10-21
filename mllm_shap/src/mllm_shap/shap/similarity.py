# pylint: disable=too-few-public-methods

"""Embedding similarity calculations for SHAP explanations."""


from typing import cast

from torch import Tensor

from ._base.similarity import BaseEmbeddingSimilarity


class CosineSimilarity(BaseEmbeddingSimilarity):
    """Cosine similarity calculation."""

    def __call__(self, base_emb: Tensor, other_embs: Tensor) -> Tensor:
        # normalize embeddings
        base_emb_norm = base_emb / base_emb.norm(dim=-1, keepdim=True)
        other_embs_norm = other_embs / other_embs.norm(dim=-1, keepdim=True)

        return cast(Tensor, (base_emb_norm * other_embs_norm).sum(dim=-1))

# pylint: disable=too-few-public-methods

"""Embedding calculation and reduction strategies for SHAP explanations."""

from torch import Tensor
import torch

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


class OpenAiEmbedding(BaseExternalEmbedding):
    """OpenAI embedding class."""

    # TODO
    def __call__(self, responses: list[ModelResponse]) -> list[Tensor]:
        raise NotImplementedError("OpenAiEmbedding is not implemented yet.")

# pylint: disable=too-few-public-methods

"""Embedding calculation and reduction strategies for SHAP explanations."""

from torch import Tensor

from ..connectors.base.chat import BaseMllmChat
from .base.embeddings import BaseEmbeddingReducer, BaseExternalEmbedding


class ZeroReducer(BaseEmbeddingReducer):
    """Dummy reducer that returns embeddings unchanged."""

    def __call__(self, embeddings: Tensor) -> Tensor:
        embeddings = super().__call__(embeddings)
        return embeddings


class MeanReducer(BaseEmbeddingReducer):
    """Reducer that computes the mean of embeddings."""

    def __call__(self, embeddings: Tensor) -> Tensor:
        embeddings = super().__call__(embeddings)
        return embeddings.mean(dim=0)


class MaxReducer(BaseEmbeddingReducer):
    """Reducer that computes the max of embeddings."""

    def __call__(self, embeddings: Tensor) -> Tensor:
        embeddings = super().__call__(embeddings)
        return embeddings.max(dim=0).values


class MinReducer(BaseEmbeddingReducer):
    """Reducer that computes the min of embeddings."""

    def __call__(self, embeddings: Tensor) -> Tensor:
        embeddings = super().__call__(embeddings)
        return embeddings.min(dim=0).values


class SumReducer(BaseEmbeddingReducer):
    """Reducer that computes the sum of embeddings."""

    def __call__(self, embeddings: Tensor) -> Tensor:
        embeddings = super().__call__(embeddings)
        return embeddings.sum(dim=0)


class FirstReducer(BaseEmbeddingReducer):
    """
    Reducer that selects the first embedding.

    :attr:`n` parameter is ignored in this reducer.
    """

    def __call__(self, embeddings: Tensor) -> Tensor:
        return embeddings[0]


class OpenAiEmbedding(BaseExternalEmbedding):
    """OpenAI embedding class."""

    # TODO
    def __call__(self, chat: BaseMllmChat) -> Tensor:
        """
        Get the external embeddings for the given chat.

        Args:
            chat: The chat instance.
        Returns:
            The external embeddings tensor.
        """
        raise NotImplementedError("OpenAiEmbedding is not implemented yet.")

"""Base class for embedding calculation reduction strategies."""

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from ...connectors.base.model_response import ModelResponse


# pylint: disable=too-few-public-methods
class BaseEmbeddingReducer(ABC):
    """Base class for embedding reduction strategies."""

    n: int | None
    """Maximum number of embeddings to sample before reduction. None means no sampling."""

    def __init__(self, n: int | None = None):
        """
        Initialize the BaseEmbeddingReducer.

        Args:
            n: Maximum number of embeddings to sample before reduction. None means no sampling.
        Raises:
            ValueError: If n is not None or a positive integer.
        """
        if not (n is None or n > 0):
            raise ValueError("n must be None or a positive integer.")
        self.n = n

    def _prepare(self, embeddings: list[Tensor]) -> list[Tensor]:
        """
        Prepare the embeddings for reduction.

        Args:
            embeddings: The input embeddings to be reduced of size (N, d, k),
                where n is number of samples, d is single embedding vector size,
                k is number of vectors per sample.
        Returns:
            The prepared embeddings.
        Raises:
            ValueError: If any embedding is not a Tensor.
        """
        for i, emb in enumerate(embeddings):
            if not isinstance(emb, Tensor):
                raise ValueError(f"Embedding at index {i} is not a Tensor.")
            if self.n is None or self.n >= emb.shape[-1]:
                continue

            indices = torch.randperm(emb.shape[0])[: self.n]
            embeddings[i] = emb[..., indices]

        return embeddings

    @abstractmethod
    def __call__(self, embeddings: list[Tensor]) -> Tensor:
        """
        Reduce the embeddings according to the specific strategy.

        Args:
            embeddings: The input embeddings to be reduced of size (N, d, k),
                where n is number of samples, d is single embedding vector size,
                k is number of vectors per sample.
        Returns:
            The reduced embeddings of size (N, d)
        """


class BaseExternalEmbedding(ABC):
    """Base class for external embeddings."""

    @abstractmethod
    def __call__(self, responses: list[ModelResponse]) -> list[Tensor]:
        """
        Get the external embeddings for the given chat.

        Args:
            responses: The model responses to get embeddings for.
        Returns:
            The external embeddings for the text and audio tokens.
        """

"""Base class for embedding calculation reduction strategies."""

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from ...connectors.base.chat import BaseMllmChat


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

    @abstractmethod
    def __call__(self, embeddings: Tensor) -> Tensor:
        """
        Reduce the embeddings according to the specific strategy.

        Args:
            embeddings: The input embeddings to be reduced.
        Returns:
            The reduced embeddings.
        """
        if self.n is not None:
            # sample n embeddings if n is specified
            indices = torch.randperm(embeddings.shape[0])[: self.n]
            embeddings = embeddings[indices]
        return embeddings


class BaseExternalEmbedding(ABC):
    """Base class for external embeddings."""

    @abstractmethod
    def __call__(self, chat: BaseMllmChat) -> Tensor:
        """
        Get the external embeddings for the given chat.

        Args:
            chat: The chat instance.
        Returns:
            The external embeddings tensor.
        """

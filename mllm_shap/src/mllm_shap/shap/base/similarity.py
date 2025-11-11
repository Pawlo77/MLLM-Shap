"""Base class for embedding similarity calculations."""

from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor


# pylint: disable=too-few-public-methods
class BaseEmbeddingSimilarity(ABC):
    """Base class for embedding similarity calculations."""

    operates_on_embeddings: bool = True
    """
    Indicates that the similarity operates on embeddings.
    If False, it operates on raw tokens.

    Used to resolve input to :func:`__call__`.
    """

    @abstractmethod
    def __call__(self, base: Any, other: Any) -> Tensor:
        """
        Compute similarity between two embeddings.

        Args:
            base: Base object.
            other: Other objects to compare against the base.
        Returns:
            Similarity scores.
        """

"""Base class for embedding similarity calculations."""

from abc import ABC, abstractmethod

from torch import Tensor


# pylint: disable=too-few-public-methods
class BaseEmbeddingSimilarity(ABC):
    """Base class for embedding similarity calculations."""

    @abstractmethod
    def __call__(self, base_emb: Tensor, other_embs: Tensor) -> Tensor:
        """
        Compute similarity between two embeddings.

        Args:
            base_emb: Base embedding tensor.
            other_embs: Other embedding tensors to compare against.
        Returns:
            Similarity scores.
        """

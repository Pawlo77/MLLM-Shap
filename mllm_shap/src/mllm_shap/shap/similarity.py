# pylint: disable=too-few-public-methods
"""Embedding similarity calculations for SHAP explanations."""

import hashlib
from typing import cast

import torch
from torch import Tensor
from sklearn.feature_extraction.text import TfidfVectorizer

from ..connectors.base.model_response import ModelResponse
from .base.similarity import BaseEmbeddingSimilarity


class EuclideanSimilarity(BaseEmbeddingSimilarity):
    """
    Euclidean similarity calculation,
    used in implementation of U4 utility function from the paper.
    """

    def __call__(self, base: Tensor, other: Tensor) -> Tensor:
        """
        Calculate the Euclidean similarity between the base embedding and other embeddings.

        Args:
            base: The base embedding tensor, shape [embedding_dim].
            other: The other embeddings tensor to compare against, shape [num_embeddings, embedding_dim].
        """
        # calculate euclidean distances
        distances = torch.norm(other - base.unsqueeze(0), dim=-1)
        # convert distances to similarities
        similarities = 1 / (1 + distances)
        return cast(Tensor, similarities)


class CosineSimilarity(BaseEmbeddingSimilarity):
    """
    Cosine similarity calculation,
    used in implementation of U1 and U2 utility functions from the paper.
    """

    def __call__(self, base: Tensor, other: Tensor) -> Tensor:
        """
        Calculate the Cosine similarity between the base embedding and other embeddings.

        Args:
            base: The base embedding tensor, shape [embedding_dim].
            other: The other embeddings tensor to compare against, shape [num_embeddings, embedding_dim].
        """
        # normalize embeddings with epsilon to avoid division by zero
        eps = 1e-8
        base_norm_val = base.norm(dim=-1, keepdim=True).clamp(min=eps)
        other_norm_val = other.norm(dim=-1, keepdim=True).clamp(min=eps)

        base_norm = base / base_norm_val
        other_norm = other / other_norm_val

        return cast(Tensor, (other_norm * base_norm.unsqueeze(0)).sum(dim=-1))


class TfIdfCosineSimilarity(BaseEmbeddingSimilarity):
    """
    TF-IDF weighted Cosine similarity calculation,
    used in implementation of U3 utility function from the paper.
    """

    operates_on_embeddings: bool = False

    __tokenize_map: dict[bytes, int] = {}
    __tokenize_counter: int = 0
    __vectorizer: TfidfVectorizer

    def __init__(self) -> None:
        """Initialize the TF-IDF vectorizer."""
        self.__vectorizer = TfidfVectorizer(analyzer=lambda x: x)

    def __call__(self, base: ModelResponse, other: list[ModelResponse]) -> Tensor:
        """
        Calculate the TF-IDF weighted Cosine similarity between the base response and other responses.

        Args:
            base: The base model response.
            other: The list of other model responses to compare against.
        Returns:
            A tensor containing the TF-IDF weighted Cosine similarities.
        """

        # check if other[0] == base
        if not (
            torch.equal(base.generated_text_tokens, other[0].generated_text_tokens)
            and torch.equal(base.generated_audio_tokens, other[0].generated_audio_tokens)
        ):
            raise ValueError("The first element of 'other' must be equal to 'base' tensor.")

        generated_text_tokens_hashes = [self.__tokenize(tensor=o.generated_text_tokens) for o in other]
        generated_audio_tokens_hashes = [self.__tokenize(tensor=o.generated_audio_tokens) for o in other]

        token_hashes_tensors = [
            torch.cat((text_hash, audio_hash), dim=0)
            for text_hash, audio_hash in zip(generated_text_tokens_hashes, generated_audio_tokens_hashes)
        ]

        tf_idfs = Tensor(
            self.__vectorizer.fit_transform([o.numpy() for o in token_hashes_tensors]).toarray(),
        ).to(base.generated_text_tokens.device)

        return CosineSimilarity()(base=tf_idfs[0], other=tf_idfs)

    def __tokenize(self, tensor: Tensor) -> Tensor:
        """
        Tokenize the input tensor into a unique integer representation.

        Args:
            tensor: The input tensor to tokenize.
        Returns:
            A tensor containing the unique integer representation of the input tensor.
        """

        tensor_cpu = tensor.detach().to("cpu").contiguous()

        result = []
        for el in tensor_cpu:
            # Hash both data and metadata
            h = hashlib.sha256()
            h.update(el.numpy().tobytes())
            h.update(str(tuple(el.shape)).encode())
            h.update(str(tensor_cpu.dtype).encode())
            key = h.digest()
            # assign incremental id if unseen
            if key not in self.__tokenize_map:
                self.__tokenize_map[key] = self.__tokenize_counter
                self.__tokenize_counter += 1
            result.append(self.__tokenize_map[key])
        return Tensor(result)

"""Validators for SHAP base modules."""

from pydantic import BaseModel, ConfigDict, model_validator

from ...connectors.base.chat import BaseMllmChat
from ...connectors.base.model import BaseMllmModel
from ..enums import Mode
from .embeddings import BaseEmbeddingReducer, BaseExternalEmbedding
from .normalizers import BaseNormalizer
from .similarity import BaseEmbeddingSimilarity
from ...connectors.base.model_response import ModelResponse


class BaseShapConfig(BaseModel):
    """
    Configuration model for BaseShap.
    Used just for validation and type checking.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    mode: Mode
    """SHAP execution mode controlling how attribution values are computed."""
    embedding_model: BaseExternalEmbedding | None
    """Optional external embedding model used before similarity evaluation."""
    embedding_reducer: BaseEmbeddingReducer
    """Reducer that maps token-level embeddings to the comparison representation."""
    similarity_measure: BaseEmbeddingSimilarity
    """Similarity function applied between reduced embeddings."""
    normalizer: BaseNormalizer
    """Normalizer applied to raw SHAP scores before returning results."""
    allow_mask_duplicates: bool
    """Whether duplicate masks are allowed during sampling/exploration."""


class BaseShapCallConfig(BaseModel):
    """
    Configuration model for BaseShap.__call__ method.
    Used just for validation and type checking.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: BaseMllmModel
    """Model connector used for response generation and embedding extraction."""
    source_chat: BaseMllmChat
    """Original chat state from which perturbations/masks are derived."""
    response: ModelResponse
    """Reference model response for the unmasked input."""
    progress_bar: bool
    """Whether to display progress information during SHAP computation."""
    verbose: bool
    """Whether to enable verbose diagnostic output while running SHAP."""

    @model_validator(mode="after")
    def check_same_chat_device(self) -> "BaseShapCallConfig":
        """
        Ensure all chat instances use the same device.
        Compares the 'device' attribute on each chat (uses None if missing).
        """
        src_dev = getattr(self.source_chat, "device", None)
        full_dev = getattr(self.response.chat, "device", None)

        if not src_dev == full_dev:
            raise ValueError(
                f"All chat instances must have the same device. "
                f"Got source={src_dev}, full={full_dev}"
            )
        return self

    @model_validator(mode="after")
    def check_response_has_chat(self) -> "BaseShapCallConfig":
        """
        Ensure the response has a chat instance.
        """
        if self.response.chat is None:
            raise ValueError("Response must have a chat instance.")
        return self

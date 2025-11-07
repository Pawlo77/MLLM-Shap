"""Validators for SHAP base modules."""

from pydantic import BaseModel, ConfigDict, model_validator

from ...connectors.base.chat import BaseMllmChat
from ...connectors.base.model import BaseMllmModel
from ..enums import Mode
from .embeddings import BaseEmbeddingReducer, BaseExternalEmbedding
from .normalizers import BaseNormalizer
from .similarity import BaseEmbeddingSimilarity
from ...connectors.base.model_response import ModelResponse


# duplicates with shap/_explainers/explainer.py
# pylint: disable=duplicate-code,too-few-public-methods
class BaseShapConfig(BaseModel):
    """
    Configuration model for BaseShap.
    Used just for validation and type checking.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    mode: Mode
    embedding_model: BaseExternalEmbedding | None
    embedding_reducer: BaseEmbeddingReducer
    similarity_measure: BaseEmbeddingSimilarity
    normalizer: BaseNormalizer
    allow_mask_duplicates: bool


# pylint: disable=too-few-public-methods
class BaseShapCallConfig(BaseModel):
    """
    Configuration model for BaseShap.__call__ method.
    Used just for validation and type checking.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: BaseMllmModel
    source_chat: BaseMllmChat
    response: ModelResponse
    progress_bar: bool
    verbose: bool

    @model_validator(mode="after")
    def check_same_chat_device(self) -> "BaseShapCallConfig":
        """
        Ensure all chat instances use the same device.
        Compares the 'device' attribute on each chat (uses None if missing).
        """
        src_dev = getattr(self.source_chat, "device", None)
        full_dev = getattr(self.response.chat, "device", None)

        if not src_dev == full_dev:
            raise ValueError(f"All chat instances must have the same device. " f"Got source={src_dev}, full={full_dev}")
        return self

    @model_validator(mode="after")
    def check_response_has_chat(self) -> "BaseShapCallConfig":
        """
        Ensure the response has a chat instance.
        """
        if self.response.chat is None:
            raise ValueError("Response must have a chat instance.")
        return self

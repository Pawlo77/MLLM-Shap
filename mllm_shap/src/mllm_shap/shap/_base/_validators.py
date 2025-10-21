"""Validators for SHAP base modules."""

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict

from ...connectors._base.chat import BaseChat
from ...connectors._base.model import BaseModel
from ..enums import Mode
from .embeddings import BaseEmbeddingReducer, BaseExternalEmbedding
from .normalizers import BaseNormalizer
from .similarity import BaseEmbeddingSimilarity


# duplicates with shap/_explainers/explainer.py
# pylint: disable=duplicate-code
class BaseShapConfig(PydanticBaseModel):
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


class BaseShapCallConfig(PydanticBaseModel):
    """
    Configuration model for BaseShap.__call__ method.
    Used just for validation and type checking.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: BaseModel
    source_chat: BaseChat
    response_chat: BaseChat
    full_chat: BaseChat
    progress_bar: bool
    verbose: bool

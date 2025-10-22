"""Validators for SHAP base modules."""

from pydantic import BaseModel
from pydantic import ConfigDict

from ...connectors.base.chat import BaseMllmChat
from ...connectors.base.model import BaseMllmModel
from ..enums import Mode
from .embeddings import BaseEmbeddingReducer, BaseExternalEmbedding
from .normalizers import BaseNormalizer
from .similarity import BaseEmbeddingSimilarity


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


# pylint: disable=too-few-public-methods
class BaseShapCallConfig(BaseModel):
    """
    Configuration model for BaseShap.__call__ method.
    Used just for validation and type checking.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: BaseMllmModel
    source_chat: BaseMllmChat
    response_chat: BaseMllmChat
    full_chat: BaseMllmChat
    progress_bar: bool
    verbose: bool

"""Base class for SHAP-based explanations."""

from abc import ABC, abstractmethod
from logging import Logger
from typing import Any

from pydantic import BaseModel, ConfigDict

from ...connectors.base.chat import BaseMllmChat
from ...connectors.base.model import BaseMllmModel
from ...utils.logger import get_logger
from ..explainer_result import ExplainerResult
from .shap_explainer import BaseShapExplainer

logger: Logger = get_logger(__name__)


# pylint: disable=too-few-public-methods
class _ExplainerConfig(BaseModel):
    """
    Configuration model for Explainer.
    Used just for validation and type checking.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    shap_explainer: BaseShapExplainer
    model: BaseMllmModel


class BaseExplainer(ABC):
    """Convenience base client for SHAP explainers."""

    shap_explainer: BaseShapExplainer
    """The SHAP explainer instance."""

    model: BaseMllmModel
    """The model connector instance."""

    total_n_calls: int = 0
    """Total number of MLLM calls made for last explanation."""

    def __init__(
        self,
        model: BaseMllmModel,
        shap_explainer: BaseShapExplainer,
    ) -> None:
        """
        Initialize the explainer.

        Args:
            model: The model connector instance.
            shap_explainer: The SHAP explainer instance.
        """
        # validation
        __config = _ExplainerConfig(
            shap_explainer=shap_explainer,
            model=model,
        )

        self.shap_explainer = __config.shap_explainer
        self.model = __config.model

    # pylint: disable=magic-value-comparison
    @abstractmethod
    def __call__(  # type: ignore[return]
        self,
        *_: Any,
        chat: BaseMllmChat,
        generation_kwargs: dict[str, Any] | None = None,
        **explanation_kwargs: Any,
    ) -> ExplainerResult:
        """
        Call the explainer - generate full response from :attr:`chat`
        using :attr:`model`, and then explain it using :attr:`shap_explainer`.

        Args:
            chat: The chat instance.
            generation_kwargs: The generation kwargs for the model.generate method.
            explanation_kwargs: The explanation kwargs for the SHAP explainer. Should not contain
                duplicate keys with generation_kwargs.
        Returns:
            The ExplainerResult instance.
        Raises:
            ValueError: If generation_kwargs or explanation_kwargs contain invalid keys or duplicate keys.
        """
        generation_kwargs = generation_kwargs or {}
        if "chat" in generation_kwargs or "keep_history" in generation_kwargs:
            raise ValueError("generation_kwargs should not contain 'chat' or 'keep_history' keys.")
        if "chat" in explanation_kwargs or "base_chat" in explanation_kwargs or "model" in explanation_kwargs:
            raise ValueError("explanation_kwargs should not contain 'chat', 'base_chat' or 'model' keys.")

        # ensure there are no duplicate keys between generation_kwargs and explanation_kwargs
        common_keys = set(generation_kwargs.keys()) & set(explanation_kwargs.keys())
        if common_keys:
            raise ValueError(f"Duplicate keys found in generation_kwargs and explanation_kwargs: {sorted(common_keys)}")

        self.total_n_calls = 0

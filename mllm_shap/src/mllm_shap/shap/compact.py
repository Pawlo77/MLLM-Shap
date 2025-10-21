"""Compact SHAP explainer implementation."""

from logging import Logger
from time import time
from typing import Any

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict
from torch import Tensor

from ..connectors._base.chat import BaseChat
from ..connectors._base.model import BaseModel
from ..utils.logger import get_logger
from ._base.explainer import BaseSHAPExplainer
from .precise import PreciseSHAPExplainer

logger: Logger = get_logger(__name__)


class _ExplainerConfig(PydanticBaseModel):
    """
    Configuration model for Explainer.
    Used just for validation and type checking.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    shap_explainer: BaseSHAPExplainer
    model: BaseModel


class ExplainerResult(PydanticBaseModel):
    """
    Result model for Explainer.

    Fields:
        full_chat: The full chat instance after generation (entire conversation).
            It will be set with SHAP values and cache.
        response_chat: The response chat instance after generation (last model response).
        source_chat: Chat to get explained (without base response).
        history: The history of chats and masks used during explanation (if applicable,
            that is if explainer was called with verbose=True).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    full_chat: BaseChat
    response_chat: BaseChat
    source_chat: BaseChat

    history: list[tuple[Tensor, BaseChat, BaseChat, BaseChat, Tensor] | None] | None


# pylint: disable=too-few-public-methods
class Explainer:
    """
    SHAP explainer for audio models.

    Fields:
        shap_explainer: The SHAP explainer instance.
        model: The model connector instance.
    """

    shap_explainer: BaseSHAPExplainer
    model: BaseModel

    def __init__(
        self,
        model: BaseModel,
        shap_explainer: BaseSHAPExplainer | None = None,
    ) -> None:
        """
        Initialize the explainer.

        Args:
            model: The model connector instance.
            shap_explainer: The SHAP explainer instance.
        """
        # validation
        __config = _ExplainerConfig(
            shap_explainer=shap_explainer or PreciseSHAPExplainer(),
            model=model,
        )

        self.shap_explainer = __config.shap_explainer
        self.model = __config.model

    # pylint: disable=magic-value-comparison
    def __call__(
        self,
        *_: Any,
        chat: BaseChat,
        generation_kwargs: dict[str, Any] | None = None,
        **explanation_kwargs: Any,
    ) -> ExplainerResult:
        """
        Call the explainer. Will overwrite internal state.

        Args:
            chat: The chat instance.
            generation_kwargs: The generation kwargs for the model.generate method.
            explanation_kwargs: The explanation kwargs for the SHAP explainer. Shoul not contain
                duplicate keys with generation_kwargs.
        Returns:
            The ExplainerResult instance.
        Raises:
            ValueError: If generation_kwargs or explanation_kwargs contain invalid keys.
        """
        generation_kwargs = generation_kwargs or {}
        if "chat" in generation_kwargs or "keep_history" in generation_kwargs:
            raise ValueError("generation_kwargs should not contain 'chat' or 'keep_history' keys.")
        if "chat" in explanation_kwargs or "base_chat" in explanation_kwargs or "model" in explanation_kwargs:
            raise ValueError("explanation_kwargs should not contain 'chat', 'base_chat' or 'model' keys.")

        t0 = time()
        logger.info("Generating full response from the model...")
        full_chat, response_chat = self.model.generate(
            chat=chat,
            keep_history=True,
            **generation_kwargs,
        )  # type: ignore[misc]
        logger.debug("Generation took %.2f seconds.", time() - t0)

        t0 = time()
        history = self.shap_explainer(
            source_chat=chat,
            response_chat=response_chat,
            full_chat=full_chat,
            model=self.model,
            **explanation_kwargs,
            **generation_kwargs,
        )
        logger.debug("Explanation took %.2f seconds.", time() - t0)

        return ExplainerResult(
            source_chat=chat,
            response_chat=response_chat,
            full_chat=full_chat,
            history=history,
        )

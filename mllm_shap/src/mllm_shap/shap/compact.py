"""Compact SHAP explainer implementation."""

from logging import Logger
from time import time
from typing import Any

from pydantic import BaseModel, ConfigDict
from torch import Tensor

from ..connectors.base.chat import BaseMllmChat
from ..connectors.base.model import BaseMllmModel
from ..connectors.base.model_response import ModelResponse
from ..utils.logger import get_logger
from .base.explainer import BaseShapExplainer
from .precise import PreciseShapExplainer

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


# pylint: disable=too-few-public-methods
class ExplainerResult(BaseModel):
    """Result model for Explainer."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    """Configuration for pydantic model."""

    full_chat: BaseMllmChat
    """The full chat instance after generation (entire conversation). It will be set with SHAP values and cache."""

    source_chat: BaseMllmChat
    """Chat to get explained (without base response)."""

    history: list[tuple[Tensor, int, BaseMllmChat | None, ModelResponse]] | None
    """
    The history of chats and masks used during explanation
    (if applicable, that is if explainer was called with `verbose=True`).
    Each entry is a tuple of  (mask, mask_hash, masked_chat, model_response)
    If cache was used, masked_chat will be None.).
    """


# pylint: disable=too-few-public-methods
class Explainer:
    """SHAP explainer for audio models."""

    shap_explainer: BaseShapExplainer
    """The SHAP explainer instance."""

    model: BaseMllmModel
    """The model connector instance."""

    def __init__(
        self,
        model: BaseMllmModel,
        shap_explainer: BaseShapExplainer | None = None,
    ) -> None:
        """
        Initialize the explainer.

        Args:
            model: The model connector instance.
            shap_explainer: The SHAP explainer instance.
        """
        # validation
        __config = _ExplainerConfig(
            shap_explainer=shap_explainer or PreciseShapExplainer(),
            model=model,
        )

        self.shap_explainer = __config.shap_explainer
        self.model = __config.model

    # pylint: disable=magic-value-comparison
    def __call__(
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
        response = self.model.generate(
            chat=chat,
            keep_history=True,
            **generation_kwargs,
        )
        logger.debug("Generation took %.2f seconds.", time() - t0)

        t0 = time()
        history = self.shap_explainer(
            model=self.model,
            source_chat=chat,
            response=response,
            **explanation_kwargs,
            **generation_kwargs,
        )
        logger.debug("Explanation took %.2f seconds.", time() - t0)

        return ExplainerResult(
            source_chat=chat,
            # chat is set as generate was called with keep_history=True
            full_chat=response.chat,  # type: ignore[arg-type]
            history=history,
        )

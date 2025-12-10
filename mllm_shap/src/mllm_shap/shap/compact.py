"""Compact SHAP explainer implementation."""

from logging import Logger
from time import time
from typing import Any


from ..connectors.base.chat import BaseMllmChat
from ..utils.logger import get_logger
from .base.explainer import BaseExplainer
from .base.shap_explainer import BaseShapExplainer
from .explainer_result import ExplainerResult
from .precise import PreciseShapExplainer

logger: Logger = get_logger(__name__)


# pylint: disable=too-few-public-methods
class Explainer(BaseExplainer):
    """
    Convenience client class for SHAP explanation.

    It generates the full response from the model
    and then uses the provided SHAP explainer to compute SHAP values.

    Uses :class:`PreciseShapExplainer` as the default SHAP explainer.
    """

    def __init__(
        self,
        shap_explainer: BaseShapExplainer | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            shap_explainer=shap_explainer or PreciseShapExplainer(),
            **kwargs,
        )

    def __call__(
        self,
        *_: Any,
        chat: BaseMllmChat,
        generation_kwargs: dict[str, Any] | None = None,
        **explanation_kwargs: Any,
    ) -> ExplainerResult:
        generation_kwargs = generation_kwargs or {}

        # validation
        super().__call__(
            chat=chat,
            generation_kwargs=generation_kwargs,
            **explanation_kwargs,
        )

        t0 = time()
        logger.info("Generating full response from the model...")
        response = self.model.generate(
            chat=chat,
            keep_history=True,
            **generation_kwargs,
        )
        logger.debug("Generation took %.2f seconds.", time() - t0)

        del chat.cache  # free memory
        t0 = time()
        history = self.shap_explainer(
            model=self.model,
            source_chat=chat,
            response=response,
            **explanation_kwargs,
            **generation_kwargs,
        )
        logger.debug("Explanation took %.2f seconds.", time() - t0)

        self.total_n_calls = self.shap_explainer.total_n_calls
        return ExplainerResult(
            source_chat=chat,
            # chat is set as generate was called with keep_history=True
            full_chat=response.chat,  # type: ignore[arg-type]
            history=history,
            total_n_calls=self.shap_explainer.total_n_calls,
        )

"""Hierarchical SHAP explainer module."""

from logging import Logger
from typing import Any


from ..connectors.base.chat import BaseMllmChat
from ..utils.logger import get_logger
from .base.explainer import BaseShapExplainer, BaseExplainer
from .explainer_result import ExplainerResult
from .precise import PreciseShapExplainer

logger: Logger = get_logger(__name__)


# pylint: disable=too-few-public-methods
class HierarchicalExplainer(BaseExplainer):
    """SHAP explainer implementing hierarchical approach for speed-up."""

    k: int
    """Maximum final group size at each level."""

    def __init__(
        self,
        shap_explainer: BaseShapExplainer | None = None,
        k: int = 10,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the explainer.

        Args:
            k: Maximum final group size at each level.
            shap_explainer: The SHAP explainer instance.
            kwargs: Additional keyword arguments.
        Raises:
            ValueError: If k is less than 1 or not an integer.
        """
        super().__init__(shap_explainer=shap_explainer or PreciseShapExplainer(), **kwargs)

        if k < 1 or int(k) != k:
            raise ValueError("k must be an integer, at least 1.")
        self.k = k

    def __call__(
        self,
        *_: Any,
        chat: BaseMllmChat,
        generation_kwargs: dict[str, Any] | None = None,
        **explanation_kwargs: Any,
    ) -> ExplainerResult:
        generation_kwargs = generation_kwargs or {}
        super().__call__(
            chat=chat,
            generation_kwargs=generation_kwargs,
            **explanation_kwargs,
        )

        raise NotImplementedError("HierarchicalExplainer is not yet implemented.")

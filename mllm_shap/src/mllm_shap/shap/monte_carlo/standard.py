"""Standard Monte Carlo approximation SHAP explainer implementation."""

from ..base.approx import StandardShapApproximation
from ._base import BaseMcShapExplainer


# pylint: disable=too-few-public-methods
class StandardMcShapExplainer(BaseMcShapExplainer, StandardShapApproximation):
    """Standard Monte Carlo SHAP Explainer."""

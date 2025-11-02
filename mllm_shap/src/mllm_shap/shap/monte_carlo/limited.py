"""Limited Monte Carlo approximation SHAP explainer implementation."""

from ..base.approx import LimitedShapApproximation
from ._base import BaseMcShapExplainer


# pylint: disable=too-few-public-methods
class LimitedMcShapExplainer(BaseMcShapExplainer, LimitedShapApproximation):
    """Limited Monte Carlo SHAP implementation."""

"""Standard Complementary SHAP explainer implementation."""

from ._base import BaseComplementaryShapExplainer


# pylint: disable=too-few-public-methods
class StandardComplementaryShapExplainer(BaseComplementaryShapExplainer):
    """Standard Complementary SHAP Explainer."""

    include_minimal_masks: bool = False

"""Standard Complementary SHAP explainer implementation."""

from ._base import BaseComplementaryShapExplainer


class StandardComplementaryShapExplainer(BaseComplementaryShapExplainer):
    """Standard Complementary SHAP Explainer."""

    include_minimal_masks: bool = False

"""Standard Neyman approximation SHAP explainer implementation."""

from ._base import BaseComplementaryNeymanShapExplainer


# pylint: disable=too-few-public-methods
class StandardComplementaryNeymanShapExplainer(BaseComplementaryNeymanShapExplainer):
    """Standard Neyman SHAP Explainer."""

    use_standard_method: bool = True

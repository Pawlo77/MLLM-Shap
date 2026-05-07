"""Standard Neyman approximation SHAP explainer implementation."""

from ._base import BaseComplementaryNeymanShapExplainer


class StandardComplementaryNeymanShapExplainer(BaseComplementaryNeymanShapExplainer):
    """Standard Neyman SHAP Explainer."""

    use_standard_method: bool = True  # overwritten

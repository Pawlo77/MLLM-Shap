"""Standard Monte Carlo approximation SHAP explainer implementation."""

from ._base import BaseMcShapExplainer


class StandardMcShapExplainer(BaseMcShapExplainer):
    """Standard Monte Carlo SHAP Explainer."""

    include_minimal_masks: bool = False

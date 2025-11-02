"""
SHAP explainers module.

Default MC SHAP explainer is :class:`LimitedMcShapExplainer`.
"""

from .compact import Explainer
from .monte_carlo import LimitedMcShapExplainer as McShapExplainer
from .complementary import ComplementaryShapExplainer
from .neyman import ComplementaryNeymanShapExplainer
from .precise import PreciseShapExplainer

__all__ = [
    "PreciseShapExplainer",
    "Explainer",
    "McShapExplainer",
    "ComplementaryShapExplainer",
    "ComplementaryNeymanShapExplainer",
]

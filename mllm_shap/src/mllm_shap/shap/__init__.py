"""
SHAP explainers module.

- :class:`PreciseShapExplainer` implements the precise SHAP value computation
    using the original SHAP algorithm.
- :class:`Explainer` implements the compact SHAP explainer that optimizes
    the computation of SHAP values for large models and datasets.
- :class:`McShapExplainer` is an alias for the limited Monte Carlo SHAP explainer.
- :class:`ComplementaryShapExplainer` implements the complementary SHAP explainer
    that focuses on explaining the complementary contributions of features.
- :class:`ComplementaryNeymanShapExplainer` implements the complementary SHAP
    explainer using Neyman allocation for improved sample efficiency.
- :class:`HierarchicalExplainer` implements a hierarchical approach to SHAP
    value computation, allowing for significant speed-ups.
"""

from .compact import Explainer
from .monte_carlo import LimitedMcShapExplainer as McShapExplainer
from .complementary import ComplementaryShapExplainer
from .neyman import ComplementaryNeymanShapExplainer
from .precise import PreciseShapExplainer
from .hierarchical import HierarchicalExplainer

__all__ = [
    "PreciseShapExplainer",
    "Explainer",
    "McShapExplainer",
    "ComplementaryShapExplainer",
    "ComplementaryNeymanShapExplainer",
    "HierarchicalExplainer",
]

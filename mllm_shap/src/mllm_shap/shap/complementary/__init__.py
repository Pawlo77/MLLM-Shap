"""
Complementary SHAP explainers.

All Complementary SHAP explainers are based on approximating SHAP values
using Monte Carlo sampling techniques. In oppose to algorithms
from `mllm_shap.shap.monte_carlo`, however, they focus rely
on complementary contributions (not marginal contributions).

They differ from standard Monte Carlo methods by including
third-part-masks, where k-th pair (S, N \\ S) follows (k = 0...n-1):
- k-th token is present in S
- |S| ~ ceil(n / 3)

- :class:`LimitedComplementaryShapExplainer` implements a limited Monte Carlo sampling
    that always includes third-part-masks.
- :class:`StandardComplementaryShapExplainer` does not include third-part-masks.
"""

from .limited import LimitedComplementaryShapExplainer
from .standard import StandardComplementaryShapExplainer

__all__ = ["LimitedComplementaryShapExplainer", "StandardComplementaryShapExplainer"]

"""
Monte Carlo SHAP explainers.

All Monte Carlo SHAP explainers are based on approximating SHAP values
using Monte Carlo sampling techniques. They differ from standard
Monte Carlo methods by including first-order-omission masks,
that is masks omitting exactly one feature (parametrizable).

First-order-omission masks are masks that omit exactly one feature from the set.

- :class:`LimitedMcShapExplainer` implements a limited Monte Carlo sampling
    that always includes first-order-omission masks.
- :class:`StandardMcShapExplainer` does not include first-order-omission masks.
- :func:`approximate_budget` is a utility function to estimate the number
    of samples required to achieve a desired error bound with a specified
    confidence level using Hoeffding's inequality.
"""

from .limited import LimitedMcShapExplainer
from .standard import StandardMcShapExplainer
from .utils import approximate_budget

__all__ = ["LimitedMcShapExplainer", "StandardMcShapExplainer", "approximate_budget"]

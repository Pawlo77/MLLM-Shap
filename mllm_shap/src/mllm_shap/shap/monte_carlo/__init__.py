"""
Monte Carlo SHAP explainers.

All Monte Carlo SHAP explainers are based on approximating SHAP values
using Monte Carlo sampling techniques. They differ from standard
Monte Carlo methods by including first-order-omission masks,
that is masks omitting exactly one feature (parametrizable).

- :class:`LimitedMcShapExplainer` implements a limited Monte Carlo sampling
    approach that avoids drawing the same mask more than once, which helps
    to better cover the feature space within a limited number of samples.
- :class:`StandardMcShapExplainer` implements the standard Monte Carlo
    sampling approach, allowing for repeated masks as per true Monte Carlo
    sampling methodology.
- :func:`approximate_budget` is a utility function to estimate the number
    of samples required to achieve a desired error bound with a specified
    confidence level using Hoeffding's inequality.
"""

from .limited import LimitedMcShapExplainer
from .standard import StandardMcShapExplainer
from .utils import approximate_budget

__all__ = ["LimitedMcShapExplainer", "StandardMcShapExplainer", "approximate_budget"]

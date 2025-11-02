"""
Monte Carlo SHAP explainers.

All Monte Carlo SHAP explainers are based on approximating SHAP values
using Monte Carlo sampling techniques. They differ from standard
Monte Carlo methods by always including minimal masks (one-versus-all).

- :class:`LimitedMcShapExplainer` implements a limited Monte Carlo sampling
    approach that avoids drawing the same mask more than once, which helps
    to better cover the feature space within a limited number of samples.
- :class:`StandardMcShapExplainer` implements the standard Monte Carlo
    sampling approach, allowing for repeated masks as per true Monte Carlo
    sampling methodology.
"""

from .limited import LimitedMcShapExplainer
from .standard import StandardMcShapExplainer

__all__ = ["LimitedMcShapExplainer", "StandardMcShapExplainer"]

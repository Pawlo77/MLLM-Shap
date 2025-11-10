"""
Module for hierarchical SHAP explainer.

- :class:`HierarchicalExplainer` - Hierarchical SHAP explainer class. It
    builds a hierarchy of feature groups and computes SHAP values for each group
    level by level. Supports text-only mode with high efficiency and
    multimodal mode with moderate efficiency (that is, fixed first level
    size dependent on input structure).
"""

from .explainer import HierarchicalExplainer

__all__ = ["HierarchicalExplainer"]

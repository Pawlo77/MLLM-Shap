"""Shared runner data structures."""

from dataclasses import dataclass
from ..config import ExplainerVariant


@dataclass(frozen=True)
class ExpandedVariant:
    """A concrete materialization of a user-declared explainer variant."""

    run_slug: str
    """Unique identifier for this run, used for MLflow tracking and result association."""
    variant: ExplainerVariant
    """The original user-declared variant this was expanded from, for reference."""
    fraction: float | None
    """Fraction of samples to run, if applicable (e.g. for hierarchical variants)."""
    num_samples: int | None
    """Number of samples to run, if applicable (e.g. for MC-like variants)."""
    linear: float | None
    """Linear scaling factor, if applicable (e.g. for MC-like variants)."""
    hier_k: int | None = None
    """Number of clusters (k) for hierarchical explainers, if applicable."""
    hier_shap_type: str | None = None
    """Shap type for hierarchical explainers, if applicable (e.g. 'interventional' or 'conditional')."""
    hier_shap_num_samples: int | None = None
    """Number of samples for hierarchical explainers, if applicable."""
    hier_shap_fraction: float | None = None
    """Fraction for hierarchical explainers, if applicable."""
    hier_first_layer_type: str | None = None
    """Shap type for first layer of hierarchical explainers, if applicable."""
    hier_first_layer_num_samples: int | None = None
    """Number of samples for first layer of hierarchical explainers, if applicable."""
    hier_first_layer_fraction: float | None = None
    """Fraction for first layer of hierarchical explainers, if applicable."""
    hier_importance_min_fraction: float | None = None
    """Minimum fraction for hierarchical importance, if applicable."""
    hier_mode: str | None = None
    """Mode for hierarchical explainers, if applicable."""

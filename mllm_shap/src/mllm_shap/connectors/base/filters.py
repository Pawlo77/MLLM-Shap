"""Base classes for token filtering strategies."""

from abc import ABC

from pydantic import BaseModel


class TokenFilter(ABC, BaseModel):
    """Base class for token filtering strategies."""

    phrases_to_exclude: set[str]
    """Set of phrases to exclude from SHAP calculations."""

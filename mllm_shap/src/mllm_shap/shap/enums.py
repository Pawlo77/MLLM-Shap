"""Configuration for SHAP."""

from enum import Enum


class Mode(str, Enum):
    """Possible modes."""

    STATIC = "static"
    CONTEXTUAL = "contextual"

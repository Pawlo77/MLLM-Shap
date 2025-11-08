"""Configuration for SHAP."""

from enum import Enum


class Mode(str, Enum):
    """Possible modes."""

    STATIC = "static"
    """
    Static mode - embeddings are computed using final model response
    tokens only, therefore they do not carry contextual information.
    """

    CONTEXTUAL = "contextual"
    """
    Contextual mode - embeddings are computed using model
    internal states, therefore they carry contextual information.
    """

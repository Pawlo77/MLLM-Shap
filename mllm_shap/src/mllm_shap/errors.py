"""Custom exception hierarchy used by ``mllm_shap``.

This module keeps domain-specific errors in one place so callers can catch
stable exception families instead of generic built-ins.
"""


class MllmShapError(Exception):
    """Base type for package-specific failures."""


class ConfigurationError(MllmShapError, ValueError):
    """Raised when configuration values are invalid or inconsistent."""


class ValidationError(MllmShapError, ValueError):
    """Raised when runtime input validation fails."""


class ConnectorError(MllmShapError, RuntimeError):
    """Raised when connector or model integration fails at runtime."""


class ExplainerError(MllmShapError, RuntimeError):
    """Raised when explainer execution fails."""


class MaskError(ValidationError):
    """Raised when SHAP mask data is invalid."""


class WorkerExecutionError(ExplainerError):
    """Raised when threaded worker execution fails."""

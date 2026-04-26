"""Unit tests for package exception hierarchy."""

from mllm_shap.errors import (
    ConfigurationError,
    ConnectorError,
    ExplainerError,
    MaskError,
    MllmShapError,
    ValidationError,
    WorkerExecutionError,
)


def test_validation_errors_keep_value_error_compatibility() -> None:
    """Validation branch should still be catchable as ValueError."""
    err = MaskError("bad mask")
    assert isinstance(err, ValueError)
    assert isinstance(err, ValidationError)
    assert isinstance(err, MllmShapError)


def test_runtime_errors_keep_runtime_error_compatibility() -> None:
    """Runtime branch should still be catchable as RuntimeError."""
    connector_err = ConnectorError("connector failed")
    worker_err = WorkerExecutionError("worker failed")

    assert isinstance(connector_err, RuntimeError)
    assert isinstance(connector_err, MllmShapError)
    assert isinstance(worker_err, ExplainerError)
    assert isinstance(worker_err, RuntimeError)


def test_configuration_error_is_value_error() -> None:
    """Configuration errors should preserve ValueError contract."""
    err = ConfigurationError("bad config")
    assert isinstance(err, ValueError)
    assert isinstance(err, MllmShapError)

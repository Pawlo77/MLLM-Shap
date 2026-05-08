"""Observability primitives for SHAP pipelines."""

from .events import StageSpan, TraceEvent
from .sink import InMemoryObservabilitySink, ObservabilitySink

__all__ = [
    "TelemetryBridge",
    "TraceEvent",
    "StageSpan",
    "ObservabilitySink",
    "InMemoryObservabilitySink",
]


def __getattr__(name: str):
    """Lazily resolve optional exports to avoid circular imports."""
    if name == "TelemetryBridge":
        from .bridge import TelemetryBridge

        return TelemetryBridge
    raise AttributeError(name)

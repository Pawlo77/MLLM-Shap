"""Structured event/span models for explainability observability."""

from dataclasses import dataclass, field
from time import time
from typing import Any


@dataclass(frozen=True)
class TraceEvent:
    """Single structured event in explainability run."""

    name: str
    """The name of the event, used to identify the type of event in the observability sink.
    This should be a descriptive name that indicates the purpose of the event,
    such as 'stage_start', 'stage_end', 'metric', or 'probe_snapshot'."""
    run_id: str
    """The unique identifier for the SHAP run that this event is associated with.
    This should match the run_id used in the TelemetryBridge to allow correlation of events and spans in the observability sink."""
    ts: float = field(default_factory=time)
    """Timestamp of the event, represented as seconds since the epoch.
    This will be automatically set to the current time when the event is created, but can be overridden if needed."""
    attrs: dict[str, Any] = field(default_factory=dict)
    """Additional attributes for the event, represented as a dictionary of key-value pairs.
    This can be used to include any relevant information about the event that may be useful
    for analysis in the observability sink, such as stage names, metric keys and values, or probe snapshot details"""


@dataclass(frozen=True)
class StageSpan:
    """Structured span for stage timing."""

    run_id: str
    """The unique identifier for the SHAP run that this span is associated with.
    This should match the run_id used in the TelemetryBridge to allow correlation of events and spans in the observability sink."""
    stage: str
    """The name of the stage that this span is measuring.
    This should correspond to the name of the stage in the pipeline, such as 'SamplingStage', 'AttributionStage', or 'PostprocessingStage'."""
    elapsed_ms: float
    """The elapsed time for the stage, measured in milliseconds.
    This should represent the total time taken to execute the stage, and can be used for performance analysis in the observability sink."""
    attrs: dict[str, Any] = field(default_factory=dict)
    """Additional attributes for the span, represented as a dictionary of key-value pairs.
    This can be used to include any relevant information about the stage execution that may be useful for analysis in the observability sink."""

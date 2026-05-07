"""Observability sink interfaces and in-memory implementation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .events import StageSpan, TraceEvent


class ObservabilitySink(ABC):
    """Abstract event sink for explainability traces."""

    @abstractmethod
    def emit_event(self, event: TraceEvent) -> None:
        """Emit structured event."""

    @abstractmethod
    def emit_span(self, span: StageSpan) -> None:
        """Emit structured stage span."""


@dataclass
class InMemoryObservabilitySink(ObservabilitySink):
    """Testing/debug sink storing all events in memory."""

    events: list[TraceEvent] = field(default_factory=list)
    spans: list[StageSpan] = field(default_factory=list)

    def emit_event(self, event: TraceEvent) -> None:
        self.events.append(event)

    def emit_span(self, span: StageSpan) -> None:
        self.spans.append(span)

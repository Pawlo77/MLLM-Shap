"""Bridge between TelemetryProbe and structured observability sinks."""

from dataclasses import dataclass

from ..shap.core.telemetry import TelemetryProbe
from .events import StageSpan, TraceEvent
from .sink import ObservabilitySink


@dataclass(frozen=True)
class TelemetryBridge:
    """Dual-writes telemetry events to structured sink."""

    run_id: str
    """Unique identifier for the SHAP run, used to correlate events and spans in the observability sink."""
    sink: ObservabilitySink
    """The observability sink to which telemetry events will be emitted.
    This should be an instance of a class that implements the ObservabilitySink interface,
    such as InMemoryObservabilitySink or a custom implementation that sends events
    to an external monitoring system."""

    def custom_metric(self, key: str, value: float | int | str | bool) -> None:
        """Emit a metric update as event."""
        self.sink.emit_event(
            TraceEvent(
                name="metric",
                run_id=self.run_id,
                attrs={"key": key, "value": value},
            )
        )

    def stage_span(self, stage: str, elapsed_ms: float) -> None:
        """Emit explicit stage duration span."""
        self.sink.emit_span(
            StageSpan(
                run_id=self.run_id,
                stage=stage,
                elapsed_ms=float(elapsed_ms),
            )
        )

    @staticmethod
    def from_probe(
        probe: TelemetryProbe | None,
        run_id: str,
        sink: ObservabilitySink,
    ) -> "TelemetryBridge":
        """Create bridge and backfill known probe metrics if available."""
        bridge = TelemetryBridge(run_id=run_id, sink=sink)
        if probe is None:
            return bridge

        metrics = probe.get_metrics().to_dict()
        for area, payload in metrics.items():
            bridge.sink.emit_event(
                TraceEvent(
                    name="probe_snapshot",
                    run_id=run_id,
                    attrs={"area": area, "payload": payload},
                )
            )
        return bridge

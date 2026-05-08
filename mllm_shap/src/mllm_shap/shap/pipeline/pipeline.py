"""Generic explain pipeline executor."""

from dataclasses import dataclass
from time import perf_counter
from uuid import uuid4

from ..core.telemetry import TelemetryProbe
from ...observability.events import TraceEvent
from ...observability.sink import ObservabilitySink
from .context import ExplainContext, ExplainState
from .contracts import PipelineStage


@dataclass(frozen=True)
class ExplainPipeline:
    """Ordered stage executor for SHAP runs."""

    stages: tuple[PipelineStage, ...]
    """Ordered tuple of pipeline stages to execute. Each stage should be
    an instance of a class that implements the PipelineStage protocol,
    meaning it should have a run method that takes
    an ExplainContext, ExplainState, and optional TelemetryProbe as"""

    def run(
        self,
        context: ExplainContext,
        state: ExplainState,
        probe: TelemetryProbe | None = None,
    ) -> ExplainState:
        """Execute stages in order and return mutated state."""
        sink = context.params.get("observability_sink")
        bridge = None
        if isinstance(sink, ObservabilitySink):
            from ...observability.bridge import TelemetryBridge

            run_id = str(
                state.metadata.get("run_id") or context.params.get("run_id") or uuid4()
            )
            state.add_metadata("run_id", run_id)
            bridge = TelemetryBridge.from_probe(probe=probe, run_id=run_id, sink=sink)

        for idx, stage in enumerate(self.stages):
            stage_name = type(stage).__name__
            if bridge is not None:
                bridge.sink.emit_event(
                    TraceEvent(
                        name="stage_start",
                        run_id=bridge.run_id,
                        attrs={"stage": stage_name, "index": idx},
                    )
                )

            started = perf_counter()
            try:
                stage.run(context=context, state=state, probe=probe)
            except Exception as exc:
                elapsed_ms = (perf_counter() - started) * 1000.0
                if bridge is not None:
                    bridge.sink.emit_event(
                        TraceEvent(
                            name="stage_error",
                            run_id=bridge.run_id,
                            attrs={
                                "stage": stage_name,
                                "index": idx,
                                "elapsed_ms": elapsed_ms,
                                "error": str(exc),
                                "error_type": type(exc).__name__,
                            },
                        )
                    )
                raise

            elapsed_ms = (perf_counter() - started) * 1000.0
            if bridge is not None:
                bridge.stage_span(stage=stage_name, elapsed_ms=elapsed_ms)
                bridge.sink.emit_event(
                    TraceEvent(
                        name="stage_end",
                        run_id=bridge.run_id,
                        attrs={
                            "stage": stage_name,
                            "index": idx,
                            "elapsed_ms": elapsed_ms,
                        },
                    )
                )
        return state

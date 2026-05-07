"""Tests for composition-first explain pipeline."""

from dataclasses import dataclass
from types import MappingProxyType

import pytest

from mllm_shap.observability.sink import InMemoryObservabilitySink
from mllm_shap.shap.pipeline import ExplainContext, ExplainPipeline, ExplainState


@dataclass
class _CounterStage:
    key: str
    value: int

    def run(self, context: ExplainContext, state: ExplainState, probe=None) -> None:
        del context, probe
        state.metadata[self.key] = state.metadata.get(self.key, 0) + self.value


@dataclass
class _FailStage:
    error: Exception

    def run(self, context: ExplainContext, state: ExplainState, probe=None) -> None:
        del context, state, probe
        raise self.error


def test_pipeline_runs_stages_in_order() -> None:
    state = ExplainState()
    pipeline = ExplainPipeline(
        stages=(
            _CounterStage(key="n", value=1),
            _CounterStage(key="n", value=2),
            _CounterStage(key="n", value=3),
        )
    )

    # ExplainContext fields are not used by this test stage implementation.
    context = ExplainContext(
        model=None,  # type: ignore[arg-type]
        source_chat=None,  # type: ignore[arg-type]
        response_chat=None,  # type: ignore[arg-type]
        base_response=None,  # type: ignore[arg-type]
        device=None,  # type: ignore[arg-type]
    )

    result = pipeline.run(context=context, state=state)

    assert result.metadata["n"] == 6


def test_pipeline_emits_observability_events_and_spans() -> None:
    sink = InMemoryObservabilitySink()
    state = ExplainState()
    pipeline = ExplainPipeline(
        stages=(
            _CounterStage(key="n", value=1),
            _CounterStage(key="n", value=2),
        )
    )
    context = ExplainContext(
        model=None,  # type: ignore[arg-type]
        source_chat=None,  # type: ignore[arg-type]
        response_chat=None,  # type: ignore[arg-type]
        base_response=None,  # type: ignore[arg-type]
        device=None,  # type: ignore[arg-type]
        params=MappingProxyType({"observability_sink": sink, "run_id": "run-123"}),
    )

    _ = pipeline.run(context=context, state=state)

    assert state.metadata["run_id"] == "run-123"
    assert len(sink.spans) == 2
    assert [span.stage for span in sink.spans] == ["_CounterStage", "_CounterStage"]
    assert all(span.elapsed_ms >= 0.0 for span in sink.spans)

    start_events = [e for e in sink.events if e.name == "stage_start"]
    end_events = [e for e in sink.events if e.name == "stage_end"]
    assert len(start_events) == 2
    assert len(end_events) == 2
    assert all(e.run_id == "run-123" for e in sink.events)


def test_pipeline_emits_stage_error_event() -> None:
    sink = InMemoryObservabilitySink()
    state = ExplainState()
    pipeline = ExplainPipeline(stages=(_FailStage(error=RuntimeError("boom")),))
    context = ExplainContext(
        model=None,  # type: ignore[arg-type]
        source_chat=None,  # type: ignore[arg-type]
        response_chat=None,  # type: ignore[arg-type]
        base_response=None,  # type: ignore[arg-type]
        device=None,  # type: ignore[arg-type]
        params=MappingProxyType({"observability_sink": sink, "run_id": "run-err"}),
    )

    with pytest.raises(RuntimeError, match="boom"):
        pipeline.run(context=context, state=state)

    error_events = [e for e in sink.events if e.name == "stage_error"]
    assert len(error_events) == 1
    assert error_events[0].attrs["stage"] == "_FailStage"
    assert error_events[0].attrs["error_type"] == "RuntimeError"
    assert len(sink.spans) == 0

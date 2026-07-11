"""Tests for ResultWriter MLflow-only integration."""

from types import SimpleNamespace

import pytest

from ..src.config import ExplainerVariant
from ..src.constants import InputModality, OutputModality
from ..src.runner import ExpandedVariant, ResultWriter


@pytest.fixture
def fake_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        experiment_set_id="exp_test",
        dataset=SimpleNamespace(
            subset="main",
            split="train",
            column_mapping=SimpleNamespace(
                language="language",
                original_language="original_language",
            ),
        ),
    )


class _FakeTracker:
    active = True

    def __init__(self) -> None:
        self.metrics: list[tuple[int, dict[str, float]]] = []
        self.dicts: list[tuple[dict, str]] = []

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self.metrics.append((step, dict(metrics)))

    def log_dict(self, data: dict, artifact_file: str) -> None:
        self.dicts.append((data, artifact_file))


def test_save_sample_logs_result_to_mlflow(
    fake_cfg: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "experiments.mllm_shapx.src.runner._compute_modality_summary",
        lambda _conv: {
            "abs_sum_text": 1.0,
            "abs_sum_audio": 0.0,
            "frac_text": 1.0,
            "frac_audio": 0.0,
            "count_text_tokens": 2,
            "count_audio_segments": 0,
        },
    )
    monkeypatch.setattr(
        "experiments.mllm_shapx.src.runner._serialize_conversation", lambda _c: []
    )
    monkeypatch.setattr(
        "experiments.mllm_shapx.src.runner.serialize_result_with_audio",
        lambda **kwargs: {},
    )

    class _FC:
        def get_conversation(self) -> list:
            return []

    class _FR:
        history = None

        def __init__(self) -> None:
            self.full_chat = _FC()
            self.base_chat = self.full_chat

    tracker = _FakeTracker()
    writer = ResultWriter(tracker, fake_cfg)
    variant = ExplainerVariant(explainer_type="exact")
    run = ExpandedVariant(
        run_slug="exact",
        variant=variant,
        fraction=None,
        num_samples=None,
        linear=None,
    )
    writer.save_sample(
        row_idx=0,
        row={"language": "en", "original_language": "en"},
        result=_FR(),
        runtime_sec=1.5,
        n_calls=10,
        user_texts=["hi"],
        input_modality=InputModality.TEXT,
        output_modality=OutputModality.TEXT,
        audio_bytes_list=None,
        explainer=object(),
        run=run,
        telemetry_metrics=None,
    )

    # Verify JSON artifact was logged
    assert len(tracker.dicts) == 1
    data, path = tracker.dicts[0]
    assert path == "samples/sample_00000_result.json"
    assert data["result_schema_version"] == 2
    assert data["row_index"] == 0
    assert data["language"] == "en"

    # Verify metrics were logged
    assert len(tracker.metrics) == 1
    step, metrics = tracker.metrics[0]
    assert step == 0
    assert metrics["progress/sample_index"] == 0.0
    assert metrics["timing/runtime_sec"] == 1.5

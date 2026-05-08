"""Unit tests for base model connector validation paths."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from mllm_shap.connectors.base.model import BaseMllmModel
from mllm_shap.connectors.base.model_response import ModelResponse
from mllm_shap.connectors.config import HuggingFaceModelConfig, ModelConfig
from mllm_shap.connectors.enums import ModelHistoryTrackingMode

from ...dummy import DummyChat


class _ModelForValidation(BaseMllmModel):
    """Concrete model used to exercise BaseMllmModel base logic."""

    def __init__(self) -> None:
        super().__init__(
            config=HuggingFaceModelConfig(repo_id="dummy/model", revision="main"),
            device=torch.device("cpu"),
            processor=SimpleNamespace(),
            model=SimpleNamespace(),
        )

    def get_new_chat(self) -> DummyChat:
        return DummyChat(num_tokens=2)

    def generate(
        self,
        chat: DummyChat,
        max_new_tokens: int = 128,
        model_config: ModelConfig = ModelConfig(),
        keep_history: bool = False,
    ) -> ModelResponse:
        # run base validation branch first
        super().generate(
            chat=chat,
            max_new_tokens=max_new_tokens,
            model_config=model_config,
            keep_history=keep_history,
        )
        return ModelResponse(
            chat=chat,
            generated_text_tokens=torch.zeros(1, 1),
            generated_audio_tokens=torch.zeros(1, 1),
            generated_modality_flag=torch.zeros(1, 1),
        )

    def get_static_embeddings(
        self, responses: list[ModelResponse]
    ) -> list[torch.Tensor]:
        super().get_static_embeddings(responses)
        return [torch.zeros(1, 2) for _ in responses]

    def _get_contextual_embeddings(
        self, static_embeddings: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        return [x + 1 for x in static_embeddings]


class _BadStaticModel(_ModelForValidation):
    """Returns invalid type from get_static_embeddings to trigger base checks."""

    def get_static_embeddings(
        self, responses: list[ModelResponse]
    ) -> list[torch.Tensor]:
        del responses
        return "bad"  # type: ignore[return-value]


class _FailingContextModel(_ModelForValidation):
    """Raises in contextual embedding implementation to test wrapper errors."""

    def _get_contextual_embeddings(
        self, static_embeddings: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        del static_embeddings
        raise ValueError("boom")


class TestBaseModelValidation:
    """Validation and orchestration checks for BaseMllmModel helper logic."""

    def test_get_static_embeddings_raises_for_non_list(self) -> None:
        """get_static_embeddings should reject non-list response containers."""
        model = _ModelForValidation()
        with pytest.raises(
            ValueError, match="responses must be a list of ModelResponse"
        ):
            _ = model.get_static_embeddings(responses="bad")  # type: ignore[arg-type]

    def test_get_contextual_embeddings_raises_if_static_not_list(self) -> None:
        """get_contextual_embeddings should reject non-list static embeddings."""
        model = _BadStaticModel()
        with pytest.raises(
            ValueError, match="static_embeddings must be an instance of list"
        ):
            _ = model.get_contextual_embeddings([])

    def test_get_contextual_embeddings_raises_if_item_not_tensor(self) -> None:
        """get_contextual_embeddings should enforce Tensor items in static list."""
        model = _ModelForValidation()
        with pytest.raises(
            ValueError,
            match="Each item in static_embeddings must be an instance of Tensor",
        ):
            _ = model.get_contextual_embeddings(static_embeddings=[torch.zeros(1), "x"])  # type: ignore[list-item]

    def test_get_static_embeddings_raises_for_non_model_response_item(self) -> None:
        """get_static_embeddings should reject lists with non-ModelResponse entries."""
        model = _ModelForValidation()
        with pytest.raises(
            ValueError, match="responses must be a list of ModelResponse"
        ):
            _ = model.get_static_embeddings(responses=["bad-item"])  # type: ignore[list-item]

    def test_get_contextual_embeddings_calls_internal_impl(self) -> None:
        """Contextual embedding call should delegate to connector implementation."""
        model = _ModelForValidation()
        result = model.get_contextual_embeddings(static_embeddings=[torch.zeros(1, 2)])
        assert len(result) == 1
        assert torch.equal(result[0], torch.ones(1, 2))

    def test_get_contextual_embeddings_uses_get_static_embeddings_when_missing_input(
        self,
    ) -> None:
        """Missing static embeddings should trigger get_static_embeddings fallback path."""
        model = _ModelForValidation()
        responses = [
            ModelResponse(
                chat=None,
                generated_text_tokens=torch.zeros(1, 1),
                generated_audio_tokens=torch.zeros(1, 1),
                generated_modality_flag=torch.zeros(1, 1),
            )
        ]
        result = model.get_contextual_embeddings(responses=responses)
        assert len(result) == 1
        assert torch.equal(result[0], torch.ones(1, 2))

    def test_get_contextual_embeddings_wraps_connector_errors(self) -> None:
        """Connector implementation failures should be wrapped in RuntimeError."""
        model = _FailingContextModel()
        with pytest.raises(
            RuntimeError, match="Error occurred in connector implementation"
        ):
            _ = model.get_contextual_embeddings(static_embeddings=[torch.zeros(1, 1)])

    def test_generate_raises_for_non_positive_max_new_tokens(self) -> None:
        """Base generate validation should reject non-positive max_new_tokens."""
        model = _ModelForValidation()
        with pytest.raises(ValueError, match="max_new_tokens must be greater than 0"):
            _ = model.generate(chat=DummyChat(num_tokens=1), max_new_tokens=0)

    def test_set_chat_history_appends_and_ends_turn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_set_chat_history should append once and end the active turn."""
        model = _ModelForValidation()
        chat = DummyChat(num_tokens=0)
        calls: dict[str, Any] = {"append": 0, "end_turn": 0}

        def _append(**kwargs: Any) -> None:
            calls["append"] += 1
            assert kwargs["history_tracking_mode"] == ModelHistoryTrackingMode.TEXT

        def _end_turn() -> None:
            calls["end_turn"] += 1

        monkeypatch.setattr(chat, "append", _append)
        monkeypatch.setattr(chat, "end_turn", _end_turn)

        model._set_chat_history(
            chat=chat,
            text_tokens=torch.tensor([1, 2]),
            audio_tokens=torch.tensor([3]),
            modality_flag=torch.tensor([0, 0, 1]),
        )

        assert calls["append"] == 1
        assert calls["end_turn"] == 1

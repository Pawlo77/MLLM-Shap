"""Unit tests for mllm_shap.shap.compact module."""

from typing import Any

import pytest
import torch
from mllm_shap.connectors.base.chat import BaseMllmChat
from mllm_shap.connectors.base.model import BaseMllmModel
from mllm_shap.connectors.base.model_response import ModelResponse
from mllm_shap.shap.base.shap_explainer import BaseShapExplainer
from mllm_shap.shap.compact import Explainer, ExplainerResult
from mllm_shap.shap.base.explainer import _ExplainerConfig
from mllm_shap.shap.precise import PreciseShapExplainer
from torch import Tensor

from ..dummy import DummyChat, DummyModel, DummyShapExplainer


@pytest.fixture
def dummy_chat() -> BaseMllmChat:
    """Fixture for DummyChat instance."""
    return DummyChat()


@pytest.fixture
def dummy_model() -> BaseMllmModel:
    """Fixture for DummyModel instance."""
    return DummyModel()


@pytest.fixture
def dummy_shap() -> BaseShapExplainer:
    """Fixture for DummyShapExplainer instance."""
    return DummyShapExplainer()


@pytest.fixture
def explainer(dummy_model: BaseMllmModel, dummy_shap: BaseShapExplainer) -> Explainer:
    """Fixture for Explainer instance."""
    return Explainer(model=dummy_model, shap_explainer=dummy_shap)


class TestExplainerConfig:
    """Tests for the _ExplainerConfig validation model."""

    def test_valid_config(
        self, dummy_model: BaseMllmModel, dummy_shap: BaseShapExplainer
    ) -> None:
        """Should initialize with valid types."""
        cfg = _ExplainerConfig(model=dummy_model, shap_explainer=dummy_shap)
        assert isinstance(cfg.model, BaseMllmModel)
        assert isinstance(cfg.shap_explainer, BaseShapExplainer)

    def test_invalid_types_raise(self) -> None:
        """Should raise if invalid argument types are passed."""
        with pytest.raises(Exception):
            _ExplainerConfig(model="notamodel", shap_explainer="notexplainer")  # type: ignore[arg-type]


class TestExplainerResult:
    """Tests for the ExplainerResult dataclass."""

    def test_result_fields(self, dummy_chat: BaseMllmChat) -> None:
        """Should correctly store fields and history structure."""
        result = ExplainerResult(
            full_chat=dummy_chat,
            source_chat=dummy_chat,
            history=[
                (
                    torch.tensor([True, False]),
                    123,
                    dummy_chat,
                    ModelResponse(
                        chat=dummy_chat,
                        generated_text_tokens=torch.tensor([1]),
                        generated_audio_tokens=torch.tensor([]),
                        generated_modality_flag=torch.tensor([False]),
                    ),
                )
            ],
            total_n_calls=5,
        )

        assert result.full_chat is dummy_chat
        assert result.source_chat is dummy_chat
        assert isinstance(result.history, list)

        mask, mask_hash, chat_obj, response = result.history[0]
        assert isinstance(mask, Tensor)
        assert isinstance(mask_hash, int)
        assert isinstance(chat_obj, BaseMllmChat)
        assert isinstance(response, ModelResponse)


class TestExplainer:
    """Tests for the Explainer class."""

    def test_init_with_custom_shap_explainer(
        self, dummy_model: BaseMllmModel, dummy_shap: BaseShapExplainer
    ) -> None:
        """Should correctly initialize with custom SHAP explainer."""
        expl = Explainer(model=dummy_model, shap_explainer=dummy_shap)
        assert expl.model is dummy_model
        assert expl.shap_explainer is dummy_shap

    def test_init_with_default_shap_explainer(self, dummy_model: BaseMllmModel) -> None:
        """Should initialize with default PreciseShapExplainer if not provided."""
        expl = Explainer(model=dummy_model)
        assert isinstance(expl.shap_explainer, PreciseShapExplainer)

    def test_call_returns_explainer_result(
        self, explainer: Explainer, dummy_chat: BaseMllmChat
    ) -> None:
        """Should return ExplainerResult after successful explanation."""
        result = explainer(chat=dummy_chat)

        assert isinstance(result, ExplainerResult)
        assert isinstance(result.full_chat, BaseMllmChat)
        assert isinstance(result.source_chat, BaseMllmChat)
        assert isinstance(result.history, list) or result.history is None

    def test_invalid_generation_kwargs_raise(
        self, explainer: Explainer, dummy_chat: BaseMllmChat
    ) -> None:
        """Should raise ValueError if forbidden keys in generation_kwargs."""
        with pytest.raises(ValueError):
            explainer(chat=dummy_chat, generation_kwargs={"chat": dummy_chat})
        with pytest.raises(ValueError):
            explainer(chat=dummy_chat, generation_kwargs={"keep_history": True})

    def test_invalid_explanation_kwargs_raise(
        self, explainer: Explainer, dummy_chat: BaseMllmChat
    ) -> None:
        """Should raise ValueError if forbidden keys in explanation_kwargs."""
        with pytest.raises(ValueError):
            explainer(chat=dummy_chat, base_chat=dummy_chat)
        with pytest.raises(ValueError):
            explainer(chat=dummy_chat, model=explainer.model)

    def test_model_generate_and_explainer_are_called(
        self, dummy_chat: BaseMllmChat
    ) -> None:
        """Should call model.generate and shap_explainer.__call__ with proper arguments."""
        shap_called = {}
        model_called = {}

        class DummyShapExplainerWithTrack(DummyShapExplainer):
            def __call__(self, **kwargs):
                shap_called["called"] = kwargs
                return super().__call__(**kwargs)

        class DummyModelWithTrack(DummyModel):
            def generate(self, **kwargs):
                model_called["called"] = kwargs
                chat_arg = kwargs.get("chat", dummy_chat)
                return ModelResponse(
                    chat=chat_arg,
                    generated_text_tokens=torch.tensor([1]),
                    generated_audio_tokens=torch.tensor([]),
                    generated_modality_flag=torch.tensor([False]),
                )

        model = DummyModelWithTrack()
        expl = Explainer(model=model, shap_explainer=DummyShapExplainerWithTrack())

        result = expl(chat=dummy_chat, generation_kwargs={"arg1": 1}, verbose=True)

        assert isinstance(result, ExplainerResult)
        assert "called" in shap_called
        assert "called" in model_called
        called_chat = model_called["called"]["chat"]
        assert isinstance(called_chat, DummyChat)
        assert called_chat.input_tokens_num > 0
        assert model_called["called"]["keep_history"] is True
        assert model_called["called"]["arg1"] == 1
        assert isinstance(result.full_chat, BaseMllmChat)
        assert isinstance(result.history, list)

    def test_duplicate_keys_between_kwargs_raise_error(
        self,
        explainer: Explainer,
        dummy_chat: BaseMllmChat,
    ) -> None:
        """Should raise when generation and explanation kwargs share a key."""
        with pytest.raises(ValueError, match="Duplicate keys"):
            explainer(chat=dummy_chat, generation_kwargs={"shared": 1}, shared=2)

    def test_history_and_total_calls_propagated(self, dummy_chat: BaseMllmChat) -> None:
        """Explainer should forward kwargs, history, and call counts from SHAP explainer."""

        class TrackingShapExplainer(BaseShapExplainer):
            """Stub SHAP explainer recording call parameters."""

            def __init__(self) -> None:
                super().__init__()
                self.last_call: dict[str, Any] | None = None

            def __call__(
                self,
                model: BaseMllmModel,
                source_chat: BaseMllmChat,
                response: ModelResponse,
                progress_bar: bool = True,
                verbose: bool = False,
                n_generator_jobs: int = 1,
                **kwargs: Any,
            ) -> list[tuple[Tensor, int, BaseMllmChat | None, ModelResponse]]:
                self.last_call = {
                    "model": model,
                    "source_chat": source_chat,
                    "response": response,
                    "progress_bar": progress_bar,
                    "verbose": verbose,
                    "n_generator_jobs": n_generator_jobs,
                    **kwargs,
                }
                self.total_n_calls = 7
                history_entry = (
                    torch.ones(1, dtype=torch.bool),
                    42,
                    None,
                    response,
                )
                return [history_entry]

            def _get_num_splits(self, n: int) -> int:
                return 0

            def _get_next_split(
                self,
                n: int,
                device: torch.device,
                generated_masks_num: int,
                existing_masks: list[Tensor] | None = None,
            ) -> Tensor | None:
                return None

            def _calculate_shap_values(
                self,
                masks: Tensor,
                similarities: Tensor,
                device: torch.device,
            ) -> Tensor:
                return torch.zeros(0)

        class DummyModelAcceptingKwargs(DummyModel):
            """Dummy model variant that records generation kwargs."""

            def __init__(self) -> None:
                super().__init__()
                self.last_generate_kwargs: dict[str, Any] | None = None

            def generate(
                self,
                chat: BaseMllmChat,
                max_new_tokens: int = 128,
                model_config=None,
                keep_history: bool = False,
                **kwargs: Any,
            ) -> ModelResponse:
                self.last_generate_kwargs = {
                    "chat": chat,
                    "max_new_tokens": max_new_tokens,
                    "model_config": model_config,
                    "keep_history": keep_history,
                    **kwargs,
                }
                return super().generate(
                    chat=chat,
                    max_new_tokens=max_new_tokens,
                    model_config=model_config,
                    keep_history=keep_history,
                )

        shap = TrackingShapExplainer()
        model_with_kwargs = DummyModelAcceptingKwargs()
        expl = Explainer(model=model_with_kwargs, shap_explainer=shap)

        result = expl(
            chat=dummy_chat,
            generation_kwargs={"alpha": 1},
            beta=2,
        )

        assert shap.last_call is not None
        assert shap.last_call["model"] is model_with_kwargs
        assert shap.last_call["source_chat"] is dummy_chat
        assert "response" in shap.last_call
        assert shap.last_call["alpha"] == 1
        assert shap.last_call["beta"] == 2
        assert result.history is not None and len(result.history) == 1
        assert result.total_n_calls == 7
        assert expl.total_n_calls == 7
        assert model_with_kwargs.last_generate_kwargs is not None
        assert model_with_kwargs.last_generate_kwargs["alpha"] == 1

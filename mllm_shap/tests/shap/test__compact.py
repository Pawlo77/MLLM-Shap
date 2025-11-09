"""Unit tests for mllm_shap.shap.compact module."""

import pytest
import torch
from mllm_shap.connectors.base.chat import BaseMllmChat
from mllm_shap.connectors.base.model import BaseMllmModel
from mllm_shap.connectors.base.model_response import ModelResponse
from mllm_shap.shap.base.explainer import BaseShapExplainer
from mllm_shap.shap.compact import Explainer, ExplainerResult, _ExplainerConfig
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

    def test_valid_config(self, dummy_model: BaseMllmModel, dummy_shap: BaseShapExplainer) -> None:
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

    def test_init_with_custom_shap_explainer(self, dummy_model: BaseMllmModel, dummy_shap: BaseShapExplainer) -> None:
        """Should correctly initialize with custom SHAP explainer."""
        expl = Explainer(model=dummy_model, shap_explainer=dummy_shap)
        assert expl.model is dummy_model
        assert expl.shap_explainer is dummy_shap

    def test_init_with_default_shap_explainer(self, dummy_model: BaseMllmModel) -> None:
        """Should initialize with default PreciseShapExplainer if not provided."""
        expl = Explainer(model=dummy_model)
        assert isinstance(expl.shap_explainer, PreciseShapExplainer)

    def test_call_returns_explainer_result(self, explainer: Explainer, dummy_chat: BaseMllmChat) -> None:
        """Should return ExplainerResult after successful explanation."""
        result = explainer(chat=dummy_chat)

        assert isinstance(result, ExplainerResult)
        assert isinstance(result.full_chat, BaseMllmChat)
        assert isinstance(result.source_chat, BaseMllmChat)
        assert isinstance(result.history, list) or result.history is None

    def test_invalid_generation_kwargs_raise(self, explainer: Explainer, dummy_chat: BaseMllmChat) -> None:
        """Should raise ValueError if forbidden keys in generation_kwargs."""
        with pytest.raises(ValueError):
            explainer(chat=dummy_chat, generation_kwargs={"chat": dummy_chat})
        with pytest.raises(ValueError):
            explainer(chat=dummy_chat, generation_kwargs={"keep_history": True})

    def test_invalid_explanation_kwargs_raise(self, explainer: Explainer, dummy_chat: BaseMllmChat) -> None:
        """Should raise ValueError if forbidden keys in explanation_kwargs."""
        with pytest.raises(ValueError):
            explainer(chat=dummy_chat, base_chat=dummy_chat)
        with pytest.raises(ValueError):
            explainer(chat=dummy_chat, model=explainer.model)

    def test_model_generate_and_explainer_are_called(self, dummy_chat: BaseMllmChat) -> None:
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
                return ModelResponse(
                    chat=dummy_chat,
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
        assert isinstance(result.full_chat, BaseMllmChat)
        assert isinstance(result.history, list)

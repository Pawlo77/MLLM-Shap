"""Unit tests for BaseExplainer convenience wrapper."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from mllm_shap.shap.base.explainer import BaseExplainer
from mllm_shap.shap.explainer_result import ExplainerResult

from ...dummy import DummyChat, DummyModel, DummyShapExplainer


class ConcreteExplainer(BaseExplainer):
    """Minimal concrete implementation used for testing BaseExplainer helpers."""

    def __init__(self, model: DummyModel, shap_explainer: DummyShapExplainer) -> None:
        super().__init__(model=model, shap_explainer=shap_explainer)
        self.calls: list[tuple[DummyChat, dict[str, Any], dict[str, Any]]] = []

    def __call__(
        self, *args: Any, chat: DummyChat, generation_kwargs: dict[str, Any] | None = None, **explanation_kwargs: Any
    ) -> ExplainerResult:
        BaseExplainer.__call__(
            self,
            *args,
            chat=chat,
            generation_kwargs=generation_kwargs,
            **explanation_kwargs,
        )
        gen_copy = dict(generation_kwargs or {})
        exp_copy = dict(explanation_kwargs)
        self.calls.append((chat, gen_copy, exp_copy))
        return ExplainerResult(
            full_chat=chat,
            source_chat=chat,
            history=None,
            total_n_calls=self.total_n_calls,
        )


class TestBaseExplainer:
    """High level tests validating BaseExplainer safeguards."""

    @staticmethod
    def _create_explainer() -> ConcreteExplainer:
        return ConcreteExplainer(model=DummyModel(), shap_explainer=DummyShapExplainer())

    def test_init_validates_dependencies(self) -> None:
        """Pydantic config should reject invalid dependency types."""
        with pytest.raises(ValidationError):
            ConcreteExplainer(model=DummyModel(), shap_explainer="oops")  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            ConcreteExplainer(model="not-a-model", shap_explainer=DummyShapExplainer())  # type: ignore[arg-type]

    def test_call_rejects_chat_in_generation_kwargs(self) -> None:
        """Passing forbidden keys in generation kwargs should raise."""
        explainer = self._create_explainer()
        with pytest.raises(ValueError, match="generation_kwargs"):
            explainer(chat=DummyChat(), generation_kwargs={"chat": DummyChat()})

    def test_call_rejects_keep_history_in_generation_kwargs(self) -> None:
        """Passing keep_history in generation kwargs should raise."""
        explainer = self._create_explainer()
        with pytest.raises(ValueError, match="keep_history"):
            explainer(chat=DummyChat(), generation_kwargs={"keep_history": True})

    def test_call_rejects_reserved_keys_in_explanation_kwargs(self) -> None:
        """Reserved explanation kwargs should trigger validation error."""
        explainer = self._create_explainer()
        with pytest.raises(ValueError, match="base_chat"):
            explainer(chat=DummyChat(), base_chat=DummyChat())
        with pytest.raises(ValueError, match="model"):
            explainer(chat=DummyChat(), model=DummyModel())

    def test_call_rejects_duplicate_keys_between_kwargs(self) -> None:
        """Common keys across generation and explanation kwargs must fail."""
        explainer = self._create_explainer()
        with pytest.raises(ValueError, match="Duplicate keys"):
            explainer(
                chat=DummyChat(),
                generation_kwargs={"temperature": 0.1},
                temperature=0.2,
            )

    def test_call_resets_total_call_counter(self) -> None:
        """Base call should reset total_n_calls before execution."""
        explainer = self._create_explainer()
        explainer.total_n_calls = 7
        result = explainer(chat=DummyChat(), explanation_kwargs={"alpha": 1.0})
        assert explainer.total_n_calls == 0
        assert result.total_n_calls == 0

    def test_call_records_arguments_for_subclass(self) -> None:
        """Subclass should receive normalized kwargs after validation."""
        explainer = self._create_explainer()
        chat = DummyChat()
        generation_kwargs = {"temperature": 0.3, "top_p": 0.9}
        result = explainer(
            "unused",
            chat=chat,
            generation_kwargs=generation_kwargs,
            sample_size=4,
        )
        assert result.full_chat is chat
        recorded_chat, recorded_gen, recorded_exp = explainer.calls[-1]
        assert recorded_chat is chat
        assert recorded_gen == generation_kwargs
        assert recorded_exp == {"sample_size": 4}

    def test_call_supports_empty_kwargs(self) -> None:
        """Calling without optional kwargs should still track invocation."""
        explainer = self._create_explainer()
        result = explainer(chat=DummyChat())
        assert result.history is None
        recorded_chat, recorded_gen, recorded_exp = explainer.calls[-1]
        assert isinstance(recorded_chat, DummyChat)
        assert recorded_gen == {}
        assert recorded_exp == {}

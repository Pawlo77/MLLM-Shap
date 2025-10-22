"""Tests for the Compact SHAP Explainer."""

import pytest
import torch
from torch import Tensor
from mllm_shap.shap.compact import Explainer, ExplainerResult
from mllm_shap.shap.base.explainer import BaseShapExplainer
from mllm_shap.connectors.base.model import BaseMllmModel
from mllm_shap.connectors.base.chat import BaseMllmChat
from ..dummy import DummyChat, DummyModel, DummyShapExplainer


@pytest.fixture
def dummy_chat_instance() -> BaseMllmChat:
    """Fixture for DummyChat instance."""
    return DummyChat()


@pytest.fixture
def dummy_model_instance() -> BaseMllmModel:
    """Fixture for DummyModel instance."""
    return DummyModel()


@pytest.fixture
def dummy_shap_instance() -> BaseShapExplainer:
    """Fixture for DummyShapExplainer instance."""
    return DummyShapExplainer()


@pytest.fixture
def explainer_instance(dummy_model_instance: DummyModel, dummy_shap_instance: DummyShapExplainer) -> Explainer:
    """Fixture for Explainer instance."""
    return Explainer(model=dummy_model_instance, shap_explainer=dummy_shap_instance)


class TestExplainerResult:
    """Tests for the ExplainerResult dataclass."""

    def test_result_fields(self, explainer_instance: Explainer, dummy_chat_instance: DummyChat) -> None:
        """Test that ExplainerResult stores fields correctly."""
        dummy_chat_instance._num_tokens = 2
        dummy_chat_instance.shap_values_mask = torch.tensor([True, True], device=dummy_chat_instance.torch_device)
        result = explainer_instance(chat=dummy_chat_instance)

        assert result.full_chat is dummy_chat_instance
        assert hasattr(result, "history")
        # history may be None if verbose=False
        if result.history is not None:
            assert isinstance(result.history, list)
            for entry in result.history:
                mask, masked_source, masked_response, masked_full, embedding = entry
                assert isinstance(mask, Tensor)
                assert isinstance(masked_source, BaseMllmChat)
                assert isinstance(masked_response, BaseMllmChat)
                assert isinstance(masked_full, BaseMllmChat)
                assert isinstance(embedding, Tensor)


class TestExplainer:
    """Tests for the Explainer class."""

    def test_init_with_custom_shap_explainer(
        self, dummy_model_instance: DummyModel, dummy_shap_instance: DummyShapExplainer
    ) -> None:
        """Test initialization with a custom SHAP explainer."""
        expl = Explainer(model=dummy_model_instance, shap_explainer=dummy_shap_instance)
        assert expl.model is dummy_model_instance
        assert expl.shap_explainer is dummy_shap_instance

    def test_init_with_default_shap_explainer(self, dummy_model_instance: DummyModel) -> None:
        """Test initialization with the default SHAP explainer."""
        expl = Explainer(model=dummy_model_instance)
        from mllm_shap.shap.precise import PreciseShapExplainer

        assert isinstance(expl.shap_explainer, PreciseShapExplainer)

    def test_call_returns_explainer_result(self, explainer_instance: Explainer, dummy_chat_instance: DummyChat) -> None:
        """Test that calling the explainer returns an ExplainerResult."""
        dummy_chat_instance._num_tokens = 3
        dummy_chat_instance.shap_values_mask = torch.tensor([True, True, True], device=dummy_chat_instance.torch_device)

        result = explainer_instance(chat=dummy_chat_instance)

        assert isinstance(result, ExplainerResult)
        assert result.full_chat is dummy_chat_instance
        assert hasattr(result.full_chat, "shap")
        # SHAP values via cache
        assert result.full_chat.shap.values.shape[0] == dummy_chat_instance._num_tokens
        assert result.full_chat.shap.normalized_values.shape[0] == dummy_chat_instance._num_tokens

    def test_call_verbose_mode_returns_history(
        self, explainer_instance: Explainer, dummy_chat_instance: DummyChat
    ) -> None:
        """Test that verbose mode returns history of computations."""
        dummy_chat_instance._num_tokens = 2
        dummy_chat_instance.shap_values_mask = torch.tensor([True, True], device=dummy_chat_instance.torch_device)

        result = explainer_instance(chat=dummy_chat_instance, verbose=True)

        assert isinstance(result.history, list)
        for entry in result.history:
            if entry is None:
                continue

            mask, masked_source, masked_response, masked_full, embedding = entry
            assert isinstance(mask, torch.Tensor)
            assert isinstance(masked_source, BaseMllmChat)
            assert isinstance(masked_response, BaseMllmChat)
            assert isinstance(masked_full, BaseMllmChat)
            assert isinstance(embedding, torch.Tensor)

    def test_call_no_tokens_to_explain_raises(
        self, explainer_instance: Explainer, dummy_chat_instance: DummyChat
    ) -> None:
        """Test that an error is raised when there are no tokens to explain."""
        dummy_chat_instance._num_tokens = 0
        dummy_chat_instance.shap_values_mask = torch.tensor([], device=dummy_chat_instance.torch_device)

        import mllm_shap.shap.base.explainer as explainer_base

        with pytest.raises(explainer_base.NoTokensToExplainError):
            explainer_instance(chat=dummy_chat_instance)

    def test_shap_values_are_nan_for_unmasked_positions(
        self, explainer_instance: Explainer, dummy_chat_instance: DummyChat
    ) -> None:
        """Test that SHAP values are NaN for tokens not marked for explanation."""
        dummy_chat_instance._num_tokens = 3
        dummy_chat_instance.shap_values_mask = torch.tensor(
            [False, True, False], device=dummy_chat_instance.torch_device
        )

        result = explainer_instance(chat=dummy_chat_instance)

        shap_vals = result.full_chat.shap.values
        # First and last tokens should be NaN
        assert torch.isnan(shap_vals[0])
        assert torch.isnan(shap_vals[2])
        # Middle token should not be NaN
        assert not torch.isnan(shap_vals[1])

    def test_normalized_values_match_values_shape(
        self, explainer_instance: Explainer, dummy_chat_instance: DummyChat
    ) -> None:
        """Test that normalized SHAP values have the same shape as raw SHAP values."""
        dummy_chat_instance._num_tokens = 4
        dummy_chat_instance.shap_values_mask = torch.tensor(
            [True, True, True, True], device=dummy_chat_instance.torch_device
        )

        result = explainer_instance(chat=dummy_chat_instance)

        assert result.full_chat.shap.normalized_values.shape == result.full_chat.shap.values.shape

    def test_shap_cache_initialized(self, explainer_instance: Explainer, dummy_chat_instance: DummyChat) -> None:
        """Test that calling the explainer initializes the `.shap` cache on the chat."""
        dummy_chat_instance._num_tokens = 2
        dummy_chat_instance.shap_values_mask = torch.tensor([True, True], device=dummy_chat_instance.torch_device)
        result = explainer_instance(chat=dummy_chat_instance)

        assert hasattr(result.full_chat, "shap")
        assert result.full_chat.shap.chat is dummy_chat_instance
        assert result.full_chat.shap.calculated_by == hash(explainer_instance.shap_explainer)
        assert result.full_chat.shap.values.numel() == dummy_chat_instance._num_tokens

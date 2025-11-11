"""Unit tests for the refactored BaseShapExplainer class."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import Tensor
from copy import deepcopy
from mllm_shap.connectors.base.chat import BaseMllmChat
from mllm_shap.connectors.base.model import BaseMllmModel
from mllm_shap.connectors.base.model_response import ModelResponse
from mllm_shap.shap.base.explainer import (
    BaseShapExplainer,
    NotEnoughTokensToExplainError,
)
from mllm_shap.shap.embeddings import MeanReducer
from mllm_shap.shap.enums import Mode
from mllm_shap.shap.normalizers import PowerShiftNormalizer
from mllm_shap.shap.similarity import CosineSimilarity

from ...dummy import DummyChat, DummyModel, DummyShapExplainer


class TestBaseShapExplainer:
    """Tests for the refactored BaseShapExplainer class."""

    @staticmethod
    @pytest.fixture
    def explainer_instance() -> BaseShapExplainer:
        """Fixture for DummyShapExplainer."""
        return DummyShapExplainer()

    @staticmethod
    @pytest.fixture
    def dummy_chat_instance() -> BaseMllmChat:
        """Fixture for BaseMllmChat."""
        return DummyChat(num_tokens=3)

    @staticmethod
    @pytest.fixture
    def dummy_model_instance() -> BaseMllmModel:
        """Fixture for DummyModel."""
        return DummyModel()

    @staticmethod
    @pytest.fixture
    def dummy_response_instance(dummy_chat_instance: BaseMllmChat) -> ModelResponse:
        """Fixture for ModelResponse."""
        return ModelResponse(
            chat=dummy_chat_instance,
            generated_audio_tokens=torch.tensor([]),
            generated_text_tokens=torch.tensor([1, 2, 3]),
            generated_modality_flag=torch.ones(3, dtype=torch.bool),
        )

    def test_initialization_defaults(self, explainer_instance: BaseShapExplainer) -> None:
        """Test default initialization components."""
        expl = explainer_instance
        assert isinstance(expl.embedding_reducer, MeanReducer)
        assert isinstance(expl.similarity_measure, CosineSimilarity)
        assert isinstance(expl.normalizer, PowerShiftNormalizer)
        assert expl.mode in (Mode.STATIC, Mode.CONTEXTUAL)

    def test_hash_returns_int(self, explainer_instance: BaseShapExplainer) -> None:
        """Test that __hash__ returns an integer and is deterministic."""
        h1 = hash(explainer_instance)
        h2 = hash(explainer_instance)
        assert isinstance(h1, int)
        assert h1 == h2

    def test_get_shap_values_computation(
        self,
        explainer_instance: BaseShapExplainer,
        dummy_model_instance: BaseMllmModel,
        dummy_response_instance: ModelResponse,
        dummy_chat_instance: BaseMllmChat,
    ) -> None:
        """Test that SHAP and normalized SHAP values are computed properly."""
        masks = torch.tensor([[True, False, True], [False, True, True]])
        responses = [dummy_response_instance, dummy_response_instance]

        explainer_instance.similarity_measure.operates_on_embeddings = True
        shap_values, normalized = explainer_instance._get_shap_values(
            source_chat=dummy_chat_instance,
            model=dummy_model_instance,
            masks=masks,
            responses=responses,
            device=torch.device("cpu"),
        )

        assert shap_values.shape == (3,)
        assert normalized.shape == (3,)
        assert torch.isfinite(normalized).all()

    def test_call_returns_history(
        self,
        explainer_instance: BaseShapExplainer,
        dummy_model_instance: BaseMllmModel,
        dummy_chat_instance: BaseMllmChat,
        dummy_response_instance: ModelResponse,
    ) -> None:
        """Test __call__ returns history correctly when verbose=True."""
        history = explainer_instance(
            model=dummy_model_instance,
            source_chat=dummy_chat_instance,
            response=dummy_response_instance,
            progress_bar=False,
            verbose=True,
        )

        assert isinstance(history, list)
        assert len(history) > 0
        for el in history:
            assert len(el) == 4
            assert isinstance(el[0], Tensor)  # generated mask
            assert isinstance(el[1], int)  # mask hash
            assert isinstance(el[2], BaseMllmChat)  # chat
            assert isinstance(el[3], ModelResponse)  # model response

    @patch("mllm_shap.shap.base.explainer.MasksManager")
    @patch("mllm_shap.shap.base.explainer.CacheManager")
    @patch("mllm_shap.shap.base.explainer.generate_responses")
    def test_call_raises_not_enough_tokens(
        self,
        mock_generate_responses: MagicMock,
        mock_cache_manager: MagicMock,
        mock_masks_manager: MagicMock,
        explainer_instance: BaseShapExplainer,
        dummy_model_instance: BaseMllmModel,
        dummy_chat_instance: BaseMllmChat,
        dummy_response_instance: ModelResponse,
    ) -> None:
        """Test that NotEnoughTokensToExplainError is raised when all chats are skipped."""
        mock_masks_manager.return_value.n = 3
        mock_masks_manager.return_value.max_masks_number = 2
        mock_masks_manager.return_value.get_initial_mask.return_value = torch.tensor([True, True, True])
        mock_cache_manager.return_value.extracted_num = 0
        # simulate all chats skipped
        mock_generate_responses.return_value = (2, [])

        with pytest.raises(NotEnoughTokensToExplainError):
            explainer_instance(
                model=dummy_model_instance,
                source_chat=dummy_chat_instance,
                response=dummy_response_instance,
                progress_bar=False,
                verbose=True,
            )

    @patch("mllm_shap.shap.base.explainer.ExplainerCache.create")
    def test_save_to_cache_creates_new_cache(
        self,
        mock_create: MagicMock,
        explainer_instance: BaseShapExplainer,
        dummy_chat_instance: BaseMllmChat,
        dummy_response_instance: ModelResponse,
    ) -> None:
        """Test _save_to_cache assigns a new ExplainerCache when no existing cache."""
        responses = [dummy_response_instance]
        masks = torch.ones((2, 3), dtype=torch.bool)
        shap_values = torch.zeros(3)
        norm_values = torch.ones(3)
        explainer_instance._save_to_cache(
            chat=dummy_chat_instance,
            source_chat=deepcopy(dummy_chat_instance),
            responses=responses,
            masks=masks,
            shap_values=shap_values,
            normalized_shap_values=norm_values,
        )
        mock_create.assert_called_once()

    def test_save_to_cache_raises_when_cache_exists(
        self,
        explainer_instance: BaseShapExplainer,
        dummy_chat_instance: BaseMllmChat,
        dummy_response_instance: ModelResponse,
    ) -> None:
        """Test _save_to_cache raises ValueError when cache already exists."""
        dummy_chat_instance.cache = object()  # simulate existing cache
        responses = [dummy_response_instance]
        masks = torch.ones((2, 3), dtype=torch.bool)
        shap_values = torch.zeros(3)
        norm_values = torch.ones(3)

        with pytest.raises(ValueError):
            explainer_instance._save_to_cache(
                chat=dummy_chat_instance,
                source_chat=deepcopy(dummy_chat_instance),
                responses=responses,
                masks=masks,
                shap_values=shap_values,
                normalized_shap_values=norm_values,
            )

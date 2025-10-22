import pytest
import torch
from unittest.mock import patch
from copy import deepcopy

from mllm_shap.connectors.base.chat import AllTextTokensFilteredOutError
from mllm_shap.shap.base.explainer import (
    NoTokensToExplainError,
    NotEnoughTokensToExplainError,
)
from mllm_shap.shap.embeddings import MeanReducer
from mllm_shap.shap.similarity import CosineSimilarity
from mllm_shap.shap.normalizers import PowerShiftNormalizer
from ...dummy import DummyChat, DummyModel, DummyShapExplainer


class TestBaseShapExplainer:
    """Tests for the updated BaseShapExplainer class."""

    @staticmethod
    @pytest.fixture
    def explainer_instance() -> DummyShapExplainer:
        """Fixture for DummyShapExplainer instance."""
        return DummyShapExplainer()

    @staticmethod
    @pytest.fixture
    def dummy_chat_instance() -> DummyChat:
        """Fixture for DummyChat instance."""
        return DummyChat(num_tokens=3)

    @staticmethod
    @pytest.fixture
    def dummy_model_instance() -> DummyModel:
        """Fixture for DummyModel instance."""
        return DummyModel()

    def test_initialization_defaults(self) -> None:
        """Test default initialization components."""
        expl = DummyShapExplainer()
        assert isinstance(expl.embedding_reducer, MeanReducer)
        assert isinstance(expl.similarity_measure, CosineSimilarity)
        assert isinstance(expl.normalizer, PowerShiftNormalizer)
        assert expl.mode.name in ("STATIC", "CONTEXTUAL")
        assert expl.embedding_model is None

    def test_generate_masks_shape_and_dtype(self) -> None:
        """Test mask generation shape and dtype."""
        expl = DummyShapExplainer()
        masks = expl._generate_masks(3, device=torch.device("cpu"))
        assert masks.shape[1] == 3
        assert masks.dtype == torch.bool

    def test_deduplicate_masks_removes_duplicates(self) -> None:
        """Test that deduplication removes duplicate masks correctly."""
        expl = DummyShapExplainer()
        new_masks = torch.tensor([[True, False], [True, True]], dtype=torch.bool)
        existing_masks = torch.tensor([[True, False]], dtype=torch.bool)
        removed, extracted = expl._BaseShapExplainer__deduplicate_masks(new_masks, existing_masks)
        assert (removed == torch.tensor([0], dtype=torch.long)).all()
        assert (extracted == torch.tensor([0], dtype=torch.long)).all()

    def test_prepare_final_masks_correctness(self) -> None:
        """Test __prepare_final_masks produces correct masks."""
        expl = DummyShapExplainer()
        mask = torch.tensor([True, False, True])
        splits = torch.tensor([[True, False], [False, True]])
        final_masks = expl._BaseShapExplainer__prepare_final_masks(
            splits, target_length=3, mask=mask, device=torch.device("cpu")
        )
        assert final_masks.shape == (2, 3)
        # unmasked positions remain True
        assert final_masks[:, 1].all()
        # masked positions follow splits
        assert (final_masks[0] == torch.tensor([True, True, False])).all()
        assert (final_masks[1] == torch.tensor([False, True, True])).all()

    def test_read_cache_initializes_and_reuses(self, dummy_chat_instance: DummyChat) -> None:
        """Test __read_cache initializes and reuses cache properly."""
        expl = DummyShapExplainer()
        masks = torch.ones((2, 3), dtype=torch.bool)
        reduced_embeddings = torch.zeros((2, 3))
        full_chat = dummy_chat_instance
        # initial read (no cache)
        updated_masks, updated_embeddings, start_idx = expl._BaseShapExplainer__read_cache(
            masks, reduced_embeddings, full_chat
        )
        assert start_idx == 1
        assert updated_masks.shape == masks.shape
        assert updated_embeddings.shape == reduced_embeddings.shape

    def test_call_verbose_returns_history(
        self, explainer_instance: DummyShapExplainer, dummy_model_instance: DummyModel, dummy_chat_instance: DummyChat
    ) -> None:
        """Test __call__ returns verbose history."""
        dummy_chat_instance.shap_values_mask = torch.tensor([True, True, False])
        history = explainer_instance(
            model=dummy_model_instance,
            source_chat=dummy_chat_instance,
            response_chat=dummy_chat_instance,
            full_chat=dummy_chat_instance,
            verbose=True,
            progress_bar=False,
        )
        assert isinstance(history, list)
        for entry in history:
            if entry is not None:
                mask, masked_source, masked_response, masked_full, embeddings = entry
                assert isinstance(mask, torch.Tensor)
                assert isinstance(masked_source, DummyChat)
                assert isinstance(masked_response, DummyChat)
                assert isinstance(masked_full, DummyChat)
                assert isinstance(embeddings, torch.Tensor)

    def test_call_non_verbose_returns_none(
        self, explainer_instance: DummyShapExplainer, dummy_model_instance: DummyModel, dummy_chat_instance: DummyChat
    ) -> None:
        """Test __call__ returns None when verbose=False."""
        dummy_chat_instance.shap_values_mask = torch.tensor([True, True, False])
        result = explainer_instance(
            model=dummy_model_instance,
            source_chat=dummy_chat_instance,
            response_chat=dummy_chat_instance,
            full_chat=dummy_chat_instance,
            verbose=False,
            progress_bar=False,
        )
        assert result is None

    def test_call_raises_no_tokens_error(
        self, explainer_instance: DummyShapExplainer, dummy_model_instance: DummyModel
    ) -> None:
        """Test NoTokensToExplainError is raised if no tokens to explain."""
        chat = DummyChat(3)
        chat.shap_values_mask = torch.tensor([False, False, False])
        with pytest.raises(NoTokensToExplainError):
            explainer_instance(
                model=dummy_model_instance,
                source_chat=chat,
                response_chat=chat,
                full_chat=chat,
            )

    def test_call_raises_not_enough_tokens_error(
        self, explainer_instance: DummyShapExplainer, dummy_model_instance: DummyModel
    ) -> None:
        """Test NotEnoughTokensToExplainError is raised if not enough tokens to explain."""
        with patch.object(DummyShapExplainer, "_generate_masks", return_value=torch.empty((0, 1), dtype=torch.bool)):
            chat = DummyChat(1)
            chat.shap_values_mask = torch.tensor([True])

            with pytest.raises(NotEnoughTokensToExplainError):
                explainer_instance(
                    model=dummy_model_instance,
                    source_chat=chat,
                    response_chat=chat,
                    full_chat=chat,
                )

    def test_normalizer_applied_to_shap_values(
        self, explainer_instance: DummyShapExplainer, dummy_model_instance: DummyModel, dummy_chat_instance: DummyChat
    ) -> None:
        """Test normalizer is applied to computed SHAP values."""
        dummy_chat_instance.shap_values_mask = torch.tensor([True, True, False])
        explainer_instance(
            model=dummy_model_instance,
            source_chat=dummy_chat_instance,
            response_chat=dummy_chat_instance,
            full_chat=dummy_chat_instance,
            verbose=True,
            progress_bar=False,
        )
        cache = dummy_chat_instance.shap
        normalized_values = cache.normalized_values
        assert torch.isfinite(normalized_values[dummy_chat_instance.shap_values_mask]).all()

    def test_hash_returns_int(self, explainer_instance: DummyShapExplainer) -> None:
        """Test that __hash__ returns an integer."""
        h = hash(explainer_instance)
        assert isinstance(h, int)

    def test_call_skips_all_text_filtered_chats(
        self, explainer_instance: DummyShapExplainer, dummy_model_instance: DummyModel
    ) -> None:
        """Test that masks that result in empty chats are skipped."""
        chat = DummyChat(3)

        def raise_error(*_, **__):
            raise AllTextTokensFilteredOutError("All text tokens filtered out.")

        # patch from chat to raise AllTextTokensFilteredOutError
        with patch.object(DummyChat, "from_chat", side_effect=raise_error):
            with pytest.raises(NotEnoughTokensToExplainError):
                explainer_instance(
                    model=dummy_model_instance,
                    source_chat=chat,
                    response_chat=deepcopy(chat),
                    full_chat=deepcopy(chat),
                )

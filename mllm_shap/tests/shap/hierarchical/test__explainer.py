"""Tests for the HierarchicalExplainer module."""

import pytest
import torch
from mllm_shap.shap.hierarchical import HierarchicalExplainer
from mllm_shap.shap.hierarchical.enums import Mode
from mllm_shap.shap.precise import PreciseShapExplainer

from ...dummy import DummyModel


class DummyChat:
    """Simple mock of BaseMllmChat for testing."""

    def __init__(self) -> None:
        # True = explainable token
        self.shap_values_mask = torch.tensor(
            [True, True, False, True, True, True, False, False, True, True]
        )
        # Modalities: 0=text, 1=image, etc.
        self.tokens_modality_flag = torch.tensor([0, 0, 0, 1, 1, 1, 0, 0, 0, 0])
        # Roles: e.g., 0=system/user, 1=model
        self.token_roles = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])


class TestHierarchicalExplainerStatic:
    """Unit tests for private static utility methods."""

    def test_repeated_buckets_regular_case(self) -> None:
        """Should repeat values and trim correctly."""
        result = HierarchicalExplainer._HierarchicalExplainer__repeated_buckets(
            n=10, k=3
        )
        expected = torch.tensor([1, 1, 1, 2, 2, 2, 3, 3, 3, 4])
        assert torch.equal(result, expected)

    def test_repeated_buckets_exact_division(self) -> None:
        """If n is multiple of k, sequence ends cleanly."""
        result = HierarchicalExplainer._HierarchicalExplainer__repeated_buckets(
            n=6, k=2
        )
        expected = torch.tensor([1, 1, 2, 2, 3, 3])
        assert torch.equal(result, expected)

    def test_repeated_buckets_single_value(self) -> None:
        """Handle smallest possible n correctly."""
        result = HierarchicalExplainer._HierarchicalExplainer__repeated_buckets(
            n=1, k=5
        )
        assert torch.equal(result, torch.tensor([1]))

    def test_get_group_props_contiguous_block(self) -> None:
        """Return correct start, end, and size for contiguous True block."""
        mask = torch.tensor([False, True, True, True, False])
        start, end, n = HierarchicalExplainer._HierarchicalExplainer__get_group_props(
            mask
        )
        assert (start, end, n) == (1, 3, 3)

    def test_get_group_props_single_true(self) -> None:
        """Handle single True element correctly."""
        mask = torch.tensor([False, True, False])
        start, end, n = HierarchicalExplainer._HierarchicalExplainer__get_group_props(
            mask
        )
        assert (start, end, n) == (1, 1, 1)


class TestHierarchicalExplainerGrouping:
    """Tests for group creation and segmentation logic."""

    def test_group_ids_with_roles(self) -> None:
        """Different modalities or roles should start new groups."""
        chat = DummyChat()
        result = HierarchicalExplainer._HierarchicalExplainer__get_group_ids(
            chat, include_role=True
        )
        expected = torch.tensor([1, 1, 0, 2, 2, 3, 0, 0, 4, 4])
        assert torch.equal(result, expected)

    def test_group_ids_without_roles(self) -> None:
        """When include_role=False, role changes should not split groups."""
        chat = DummyChat()
        result = HierarchicalExplainer._HierarchicalExplainer__get_group_ids(
            chat, include_role=False
        )
        expected = torch.tensor([1, 1, 0, 2, 2, 2, 0, 0, 3, 3])
        assert torch.equal(result, expected)

    def test_empty_mask_returns_all_zero_groups(self) -> None:
        """When no tokens are explainable, result should be all zeros."""
        chat = DummyChat()
        chat.shap_values_mask = torch.zeros_like(
            chat.shap_values_mask, dtype=torch.bool
        )
        group_ids = HierarchicalExplainer._HierarchicalExplainer__get_group_ids(chat)
        assert torch.equal(
            group_ids, torch.zeros_like(chat.shap_values_mask, dtype=torch.long)
        )


class TestHierarchicalExplainerCore:
    """Tests for initialization and key numeric properties."""

    def test_get_subgroups_num(self) -> None:
        """Subgroup count should equal ceil(log(n, k))."""
        explainer = HierarchicalExplainer(k=4, shap_explainer=PreciseShapExplainer(), model=DummyModel())
        assert explainer._HierarchicalExplainer__get_subgroups_num(10) == 2
        assert explainer._HierarchicalExplainer__get_subgroups_num(8) == 2
        assert explainer._HierarchicalExplainer__get_subgroups_num(4) == 1

    def test_invalid_k_raises_valueerror(self) -> None:
        """Non-positive or non-integer k should raise ValueError."""
        with pytest.raises(ValueError):
            HierarchicalExplainer(k=0, shap_explainer=PreciseShapExplainer(), model=DummyModel())
        with pytest.raises(ValueError):
            HierarchicalExplainer(k=2.5, shap_explainer=PreciseShapExplainer(), model=DummyModel())

    def test_default_mode_is_text(self) -> None:
        """Ensure default mode is Mode.TEXT."""
        explainer = HierarchicalExplainer(k=5, shap_explainer=PreciseShapExplainer(), model=DummyModel())
        assert explainer.mode == Mode.TEXT

    def test_mode_explicit_setting(self) -> None:
        """Ensure custom mode is properly set."""
        explainer = HierarchicalExplainer(k=5, shap_explainer=PreciseShapExplainer(), model=DummyModel(), mode=Mode.MULTI_MODAL_MULTI_USER)
        assert explainer.mode == Mode.MULTI_MODAL_MULTI_USER

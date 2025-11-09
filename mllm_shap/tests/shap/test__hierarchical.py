"""Tests for the HierarchicalExplainer module."""

import pytest
import torch
from mllm_shap.shap.hierarchical import HierarchicalExplainer

from ..dummy import DummyModel


class DummyChat:
    """Simple mock of BaseMllmChat used for unit testing."""

    def __init__(self) -> None:
        self.shap_values_mask = torch.tensor([True, True, False, True, True, True, False, False, True, True])
        self.tokens_modality_flag = torch.tensor([0, 0, 0, 1, 1, 1, 0, 0, 0, 0])
        self.token_roles = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])


class TestHierarchicalExplainer:
    """Tests for __repeated_buckets static method."""

    def test_regular_case(self) -> None:
        """Check proper bucket repetition and trimming."""
        result = HierarchicalExplainer._HierarchicalExplainer__repeated_buckets(n=10, k=3)
        expected = torch.tensor([1, 1, 1, 2, 2, 2, 3, 3, 3, 4])
        assert torch.equal(result, expected)

    def test_exact_division(self) -> None:
        """Case when n is multiple of k."""
        result = HierarchicalExplainer._HierarchicalExplainer__repeated_buckets(n=6, k=2)
        expected = torch.tensor([1, 1, 2, 2, 3, 3])
        assert torch.equal(result, expected)

    def test_single_value(self) -> None:
        """Handles minimal case correctly."""
        result = HierarchicalExplainer._HierarchicalExplainer__repeated_buckets(n=1, k=5)
        expected = torch.tensor([1])
        assert torch.equal(result, expected)

    def test_contiguous_segment(self) -> None:
        """Extract start, end, and size for contiguous True block."""
        mask = torch.tensor([False, True, True, True, False])
        start, end, n = HierarchicalExplainer._HierarchicalExplainer__get_group_props(mask)
        assert (start, end, n) == (1, 3, 3)

    def test_single_true_value(self) -> None:
        """Handles single True correctly."""
        mask = torch.tensor([False, True, False])
        start, end, n = HierarchicalExplainer._HierarchicalExplainer__get_group_props(mask)
        assert (start, end, n) == (1, 1, 1)

    def test_group_splitting_by_modality_and_role(self) -> None:
        """Check that modality and role changes start new groups."""
        chat = DummyChat()
        group_ids = HierarchicalExplainer._HierarchicalExplainer__get_group_ids(chat)
        expected = torch.tensor([1, 1, 0, 2, 2, 3, 0, 0, 4, 4])
        assert torch.equal(group_ids, expected)

    def test_empty_mask(self) -> None:
        """If no explainable tokens exist, group IDs should all be zero."""
        chat = DummyChat()
        chat.shap_values_mask = torch.zeros_like(chat.shap_values_mask, dtype=torch.bool)
        group_ids = HierarchicalExplainer._HierarchicalExplainer__get_group_ids(chat)
        assert torch.equal(group_ids, torch.zeros_like(chat.shap_values_mask, dtype=torch.long))

    def test_subgroup_count_correctness(self) -> None:
        """Number of subgroups = ceil(n / k)."""
        explainer = HierarchicalExplainer(k=4, model=DummyModel())
        assert explainer._HierarchicalExplainer__get_subgroups_num(10) == 3
        assert explainer._HierarchicalExplainer__get_subgroups_num(8) == 2
        assert explainer._HierarchicalExplainer__get_subgroups_num(4) == 1

    def test_invalid_k_raises(self) -> None:
        """Ensure invalid k values raise ValueError."""
        with pytest.raises(ValueError):
            HierarchicalExplainer(k=0, model=DummyModel())
        with pytest.raises(ValueError):
            HierarchicalExplainer(k=2.5, model=DummyModel())

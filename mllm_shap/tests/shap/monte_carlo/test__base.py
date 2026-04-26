"""Unit tests for BaseMcShapExplainer class."""

import pytest
import torch
from torch import Tensor
from mllm_shap.shap.monte_carlo._base import BaseMcShapExplainer


class DummyMcExplainer(BaseMcShapExplainer):
    """Concrete implementation for testing abstract BaseMcShapExplainer."""

    def __init__(self, num_samples: int | None = None, fraction: float = 1.0) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.fraction = fraction
        self.include_minimal_masks = True
        # reset internal state
        self._first_call = True
        self._zero_mask_skipped = False
        self._base_masks = None
        self._base_calls_num = 0


class TestBaseMcShapExplainer:
    """Unit tests for BaseMcShapExplainer methods and edge cases."""

    @pytest.fixture
    def explainer(self) -> BaseMcShapExplainer:
        """Fixture for initialized DummyMcExplainer."""
        return DummyMcExplainer(num_samples=None, fraction=0.5)

    def test_get_num_splits_with_num_samples_negative_one(
        self, explainer: BaseMcShapExplainer
    ) -> None:
        """Should return n + 1 when num_samples = -1 (minimal mode)."""
        explainer.num_samples = -1
        result = explainer._get_num_splits(n=4)
        assert result == 5

    def test_get_num_splits_with_num_samples_greater_than_possible(
        self, explainer: BaseMcShapExplainer
    ) -> None:
        """Should clamp to maximum number of masks if num_samples too large."""
        explainer.num_samples = 9999
        result = explainer._get_num_splits(n=3)
        assert result == 2**3 - 1

    def test_get_num_splits_with_fraction(self, explainer: BaseMcShapExplainer) -> None:
        """Should compute number of splits based on fraction if num_samples is None."""
        explainer.num_samples = None
        explainer.fraction = 0.25
        result = explainer._get_num_splits(n=10)
        expected = int((2**10 - 1) * 0.25)
        assert result == expected

    def test_get_num_splits_fraction_rounds_down(
        self, explainer: BaseMcShapExplainer
    ) -> None:
        """Fractional budgets should be floored to the nearest integer."""
        explainer.num_samples = None
        explainer.fraction = 0.3333
        result = explainer._get_num_splits(n=10)
        expected = int((2**10 - 1) * 0.3333)
        assert result == expected

    def test_get_num_splits_cache_clear_respects_updates(
        self, explainer: BaseMcShapExplainer
    ) -> None:
        """Clearing the cache allows new sampling parameters to take effect."""
        explainer.num_samples = 6
        first = explainer._get_num_splits(n=3)
        explainer.num_samples = 4
        explainer._get_num_splits.cache_clear()  # type: ignore[attr-defined]
        second = explainer._get_num_splits(n=3)
        assert first == 6
        assert second == 4

    def test_generate_minimal_splits_shape_and_values(
        self, explainer: BaseMcShapExplainer
    ) -> None:
        """_generate_minimal_splits() should return correct shape and one-hot pattern."""
        device = torch.device("cpu")
        masks = explainer._generate_minimal_splits(n=3, device=device)
        assert masks.shape == (4, 3)
        # first row all False
        assert torch.sum(masks[0]) == 0
        # subsequent rows: each one False
        for i in range(1, 4):
            row = masks[i]
            assert torch.sum(~row) == 1
            false_index = torch.where(~row)[0].item()
            assert false_index == i - 1

    def test_get_next_split_returns_minimal_then_random(
        self, explainer: BaseMcShapExplainer
    ) -> None:
        """Should yield minimal masks first, then random masks up to budget."""
        device = torch.device("cpu")
        n = 3
        explainer.num_samples = 6  # enough for minimal + random

        # first call: minimal mask
        mask0 = explainer._get_next_split(n=n, device=device, generated_masks_num=0)
        assert mask0.shape == (n,)
        assert mask0.dtype == torch.bool

        # next minimal mask
        mask1 = explainer._get_next_split(n=n, device=device, generated_masks_num=1)
        assert mask1 is not None

        # random mask after minimal ones
        mask_random = explainer._get_next_split(
            n=n, device=device, generated_masks_num=n + 1
        )
        assert mask_random.shape == (1, n)
        assert mask_random.dtype == torch.bool

    def test_get_next_split_without_minimal_masks(
        self, explainer: BaseMcShapExplainer
    ) -> None:
        """When minimal masks are disabled, random masks should be produced immediately."""
        explainer.include_minimal_masks = False
        device = torch.device("cpu")
        n = 4
        mask = explainer._get_next_split(n=n, device=device, generated_masks_num=0)
        assert mask.shape == (1, n)
        assert mask.dtype == torch.bool

    def test_get_next_split_returns_none_when_exceeded_budget(
        self, explainer: BaseMcShapExplainer
    ) -> None:
        """Should return None if all masks already generated."""
        device = torch.device("cpu")
        explainer.num_samples = 3
        explainer.include_minimal_masks = False
        result = explainer._get_next_split(n=3, device=device, generated_masks_num=10)
        assert result is None

    def test_calculate_shap_values_computation(
        self, explainer: BaseMcShapExplainer
    ) -> None:
        """Should correctly compute included-excluded mean difference."""
        device = torch.device("cpu")
        masks = torch.tensor([[True, False], [False, True]], dtype=torch.bool)
        similarities = torch.tensor([1.0, 3.0])
        result = explainer._calculate_shap_values(
            masks=masks, similarities=similarities, device=device
        )
        assert isinstance(result, Tensor)
        # included_mean = [(1 + 0)/1, (0 + 3)/1] = [1, 3]
        # excluded_mean = [(0 + 3)/1, (1 + 0)/1] = [3, 1]
        # diff = [-2, 2]
        assert torch.allclose(result, torch.tensor([-2.0, 2.0]))

    def test_calculate_shap_values_preserves_device_and_dtype(
        self, explainer: BaseMcShapExplainer
    ) -> None:
        """The SHAP value tensor should keep the similarities dtype and device."""
        device = torch.device("cpu")
        masks = torch.tensor(
            [[True, False, True], [False, True, True], [True, True, False]],
            dtype=torch.bool,
            device=device,
        )
        similarities = torch.tensor(
            [0.5, 0.75, 1.25], dtype=torch.float64, device=device
        )
        result = explainer._calculate_shap_values(
            masks=masks, similarities=similarities, device=device
        )
        assert result.dtype == torch.float64
        assert result.device == device

    def test_calculate_shap_values_zero_difference(
        self, explainer: BaseMcShapExplainer
    ) -> None:
        """Identical similarities should lead to zero-valued SHAP contributions."""
        device = torch.device("cpu")
        masks = torch.tensor(
            [[True, False, True], [False, True, True], [True, True, False]],
            dtype=torch.bool,
        )
        similarities = torch.ones(masks.shape[0], device=device)
        result = explainer._calculate_shap_values(
            masks=masks, similarities=similarities, device=device
        )
        assert torch.allclose(result, torch.zeros(masks.shape[1], device=device))

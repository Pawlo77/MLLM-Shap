"""Unit tests for BaseMcShapExplainer class."""

import pytest
import torch
from mllm_shap.shap.monte_carlo._base import BaseMcShapExplainer
from torch import Tensor


class DummyMcExplainer(BaseMcShapExplainer):
    """Concrete implementation for testing abstract BaseMcShapExplainer."""

    def __init__(self, num_samples=None, fraction=1.0):
        super().__init__()
        self.num_samples = num_samples
        self.fraction = fraction
        self.include_minimal_masks = True
        self.__base_masks = None


class TestBaseMcShapExplainer:
    """Unit tests for BaseMcShapExplainer methods and edge cases."""

    @pytest.fixture
    def explainer(self) -> BaseMcShapExplainer:
        """Fixture for initialized DummyMcExplainer."""
        return DummyMcExplainer(num_samples=None, fraction=0.5)

    def test_get_num_splits_with_num_samples_negative_one(self, explainer: BaseMcShapExplainer) -> None:
        """Should return target_length + 1 when num_samples = -1 (minimal mode)."""
        explainer.num_samples = -1
        result = explainer._get_num_splits(target_length=4)
        assert result == 5

    def test_get_num_splits_with_num_samples_less_than_target_raises(self, explainer: BaseMcShapExplainer) -> None:
        """Should raise ValueError when num_samples < target_length."""
        explainer.num_samples = 2
        with pytest.raises(ValueError, match="num_samples must be at least"):
            _ = explainer._get_num_splits(target_length=5)

    def test_get_num_splits_with_num_samples_greater_than_possible(self, explainer: BaseMcShapExplainer) -> None:
        """Should clamp to maximum number of masks if num_samples too large."""
        explainer.num_samples = 9999
        result = explainer._get_num_splits(target_length=3)
        assert result == 2**3 - 1

    def test_get_num_splits_with_fraction(self, explainer: BaseMcShapExplainer) -> None:
        """Should compute number of splits based on fraction if num_samples is None."""
        explainer.num_samples = None
        explainer.fraction = 0.25
        result = explainer._get_num_splits(target_length=4)
        expected = int((2**4 - 1) * 0.25)
        assert result == expected

    def test_generate_minimal_splits_shape_and_values(self, explainer: BaseMcShapExplainer) -> None:
        """__generate_minimal_splits() should return correct shape and one-hot pattern."""
        device = torch.device("cpu")
        masks = explainer._BaseMcShapExplainer__generate_minimal_splits(target_length=3, device=device)
        assert masks.shape == (4, 3)
        assert masks[0].sum() == 0  # empty mask
        assert torch.equal(masks[1:], torch.eye(3, dtype=torch.bool))  # one-hot rows

    def test_get_next_split_returns_minimal_masks_then_random(self, explainer: BaseMcShapExplainer) -> None:
        """Should yield minimal masks first, then random masks up to budget."""
        device = torch.device("cpu")
        target_length = 3
        explainer.num_samples = 6  # enough for all minimal + random
        mask0 = explainer._get_next_split(target_length=target_length, device=device, generated_masks=0)
        assert mask0 is not None and mask0.dtype == torch.bool

        # next minimal masks
        mask1 = explainer._get_next_split(target_length=target_length, device=device, generated_masks=1)
        assert mask1 is not None

        # random mask after minimal ones
        mask_random = explainer._get_next_split(
            target_length=target_length,
            device=device,
            generated_masks=target_length + 1,
        )
        assert mask_random is not None and mask_random.shape == (1, target_length)

    def test_get_next_split_raises_if_no_base_masks(self, explainer: BaseMcShapExplainer) -> None:
        """Should raise RuntimeError if __base_masks is unexpectedly None."""
        device = torch.device("cpu")
        explainer._BaseMcShapExplainer__base_masks = None
        explainer.include_minimal_masks = True
        with pytest.raises(RuntimeError, match="Base masks are not present"):
            _ = explainer._get_next_split(target_length=3, device=device, generated_masks=5)

    def test_get_next_split_raises_if_sampling_budget_too_low(self, explainer: BaseMcShapExplainer) -> None:
        """Should raise RuntimeError if sampling budget smaller than minimal masks count."""
        device = torch.device("cpu")
        explainer.num_samples = 2
        explainer._BaseMcShapExplainer__base_masks = torch.zeros((5, 3), dtype=torch.bool)
        with pytest.raises(RuntimeError, match="Not enough sampling budget"):
            _ = explainer._get_next_split(target_length=2, device=device, generated_masks=0)

    def test_get_next_split_returns_none_when_exceeded_budget(self, explainer: BaseMcShapExplainer) -> None:
        """Should return None if all masks already generated."""
        device = torch.device("cpu")
        explainer.num_samples = 3
        explainer._BaseMcShapExplainer__base_masks = torch.zeros((2, 3), dtype=torch.bool)
        result = explainer._get_next_split(target_length=3, device=device, generated_masks=5)
        assert result is None

    def test_calculate_shap_values_computation(self, explainer: BaseMcShapExplainer) -> None:
        """Should correctly compute included-excluded mean difference."""
        device = torch.device("cpu")
        masks = torch.tensor([[True, False], [False, True]], dtype=torch.bool)
        similarities = torch.tensor([1.0, 3.0])
        result = explainer._calculate_shap_values(masks=masks, similarities=similarities, device=device)
        assert isinstance(result, Tensor)
        # included_mean = [(1 + 0)/1, (0 + 3)/1] = [1, 3]
        # excluded_mean = [(0 + 3)/1, (1 + 0)/1] = [3, 1]
        # diff = [-2, 2]
        assert torch.allclose(result, torch.tensor([-2.0, 2.0]))

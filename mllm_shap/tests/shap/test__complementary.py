"""Unit tests for ComplementaryShapExplainer class."""

import pytest
import torch
from mllm_shap.shap.complementary import ComplementaryShapExplainer
from mllm_shap.shap.base._masks_manager import MasksManager
from torch import Tensor
from ..dummy import DummyChat


class DummyComplementaryExplainer(ComplementaryShapExplainer):
    """Concrete implementation for testing abstract ComplementaryShapExplainer."""

    def __init__(self, num_samples: int | None = None, fraction: float = 1.0) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.fraction = fraction
        # reset internal state
        self._first_call = True
        self._zero_mask_skipped = False
        self._base_masks = None
        self._base_calls_num = 0
        self._ComplementaryShapExplainer__next_mask = None


class TestComplementaryShapExplainerNumSplits:
    """Tests for _get_num_splits logic."""

    @pytest.fixture
    def explainer(self) -> ComplementaryShapExplainer:
        """Provides a default explainer instance for testing."""
        return DummyComplementaryExplainer(num_samples=None, fraction=0.5)

    def test_num_samples_too_low_raises(self, explainer: ComplementaryShapExplainer) -> None:
        """Tests that setting num_samples too low raises ValueError."""
        explainer.num_samples = 3
        with pytest.raises(ValueError, match="num_samples must be at least"):
            _ = explainer._get_num_splits(n=2)

    def test_num_samples_odd_raises(self, explainer: ComplementaryShapExplainer) -> None:
        """Tests that setting num_samples to an odd number raises ValueError."""
        explainer.num_samples = 9
        with pytest.raises(ValueError, match="num_samples must not be odd"):
            _ = explainer._get_num_splits(n=4)

    def test_num_samples_too_large_clamps(self, explainer: ComplementaryShapExplainer) -> None:
        """Tests that num_samples too large is clamped to maximum possible."""
        explainer.num_samples = 999
        result = explainer._get_num_splits(n=3)
        assert result == 2**3 - 2

    def test_fraction_returns_even_or_adjusted(self, explainer: ComplementaryShapExplainer) -> None:
        """Tests that fraction-based sample count is computed correctly."""
        explainer.num_samples = None
        explainer.fraction = 0.6
        n = 4
        result = explainer._get_num_splits(n=n)
        total_masks = 2**n - 1
        expected = int(total_masks * explainer.fraction)
        # ensure even unless total_masks reached
        if expected % 2 == 1 and expected != total_masks:
            expected -= 1
        assert result == expected


class TestComplementaryShapExplainerNextSplit:
    """Tests for _get_next_split behavior."""

    @pytest.fixture
    def explainer(self) -> ComplementaryShapExplainer:
        """Provides a default explainer instance for testing."""
        return DummyComplementaryExplainer(num_samples=4)

    def test_complementary_mask_pair_generation(self) -> None:
        """Tests that complementary mask pairs are generated correctly."""
        explainer = DummyComplementaryExplainer(num_samples=8)
        explainer._M = None
        mask_manager = MasksManager(chat=DummyChat())
        explainer._first_call = False

        gen = explainer._get_masks_generator(
            mask_manager=mask_manager,
            device=torch.device("cpu"),
            masks=[],
        )
        mask_1, _ = next(gen)
        mask_2, _ = next(gen)

        # mask2 should be complement of mask1
        assert torch.equal(mask_2, ~mask_1)

    def test_returns_none_when_budget_exceeded(self) -> None:
        """Tests that None is returned when sample budget is exceeded."""
        explainer = DummyComplementaryExplainer(num_samples=8)
        explainer._first_call = False
        result = explainer._get_next_split(n=3, device=torch.device("cpu"), generated_masks_num=9)
        assert result is None


class TestComplementaryShapExplainerCalculateShapValues:
    """Tests for _calculate_shap_values method."""

    def test_complementary_pairs_computation(self) -> None:
        """Tests that complementary pairs are computed correctly."""
        explainer = DummyComplementaryExplainer()
        device = torch.device("cpu")
        # 2 complementary pairs
        masks = torch.tensor(
            [
                [True, True],
                [True, False],
                [False, True],
                [False, True],
                [True, False],
            ],
            dtype=torch.bool,
        )
        explainer._M = torch.tensor(
            [
                [2, 0],
                [0, 2],
            ],
            dtype=torch.float32,
            device=device,
        )
        explainer._C = torch.tensor(
            [
                [4, 0],
                [0, 6],
            ],
            dtype=torch.float32,
            device=device,
        )
        explainer._zero_mask_skipped = True
        similarities = torch.tensor([1.0, 1.0, 3.0, 2.0, 4.0])
        result = explainer._calculate_shap_values(masks=masks, similarities=similarities, device=device)
        assert isinstance(result, Tensor)
        assert result.shape[0] == masks.shape[1]

    def test_raises_for_non_complementary_pair(self) -> None:
        """Tests that ValueError is raised for non-complementary mask pairs."""
        explainer = DummyComplementaryExplainer()
        device = torch.device("cpu")
        masks = torch.tensor(
            [
                [True, True],
                [True, False],
                [True, False],
            ],
            dtype=torch.bool,
        )
        explainer._M = torch.tensor(
            [
                [2, 0],
                [0, 2],
            ],
            dtype=torch.float32,
            device=device,
        )
        explainer._C = torch.tensor(
            [
                [4, 0],
                [0, 6],
            ],
            dtype=torch.float32,
            device=device,
        )
        explainer._zero_mask_skipped = True
        similarities = torch.tensor([1.0, 1.0, 2.0])
        with pytest.raises(ValueError, match="Masks are not complementary pairs"):
            _ = explainer._calculate_shap_values(masks, similarities, device)

    def test_raises_for_odd_number_of_masks(self) -> None:
        """Tests that ValueError is raised for odd number of masks."""
        explainer = DummyComplementaryExplainer()
        device = torch.device("cpu")
        masks = torch.tensor(
            [
                [True, True],
                [True, False],
                [False, True],
                [True, True],
            ],
            dtype=torch.bool,
        )
        explainer._M = torch.tensor(
            [
                [2, 0],
                [0, 2],
            ],
            dtype=torch.float32,
            device=device,
        )
        explainer._C = torch.tensor(
            [
                [4, 0],
                [0, 6],
            ],
            dtype=torch.float32,
            device=device,
        )
        explainer._zero_mask_skipped = True
        similarities = torch.tensor([1.0, 1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Masks should be in complementary pairs"):
            _ = explainer._calculate_shap_values(masks, similarities, device)

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
        super().__init__(num_samples=num_samples, fraction=fraction)
        # reset internal state
        self._first_call = True
        self._zero_mask_skipped = False
        self._base_masks = None
        self._base_calls_num = 0
        self._ComplementaryShapExplainer__next_mask = None
        self._M = None
        self._C = None


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

    def test_fraction_respects_validation(self) -> None:
        """_validate_sampling_params enforces fraction constraints."""
        with pytest.raises(ValueError, match="fraction must be a float in the range"):
            DummyComplementaryExplainer(num_samples=None, fraction=-0.1)

    def test_num_splits_cache_depends_on_configuration(self) -> None:
        """Different sampling settings should produce different cached values."""
        explainer = DummyComplementaryExplainer(num_samples=None, fraction=0.5)
        first = explainer._get_num_splits(n=3)
        other = DummyComplementaryExplainer(num_samples=6)
        second = other._get_num_splits(n=3)
        assert first != second


class TestComplementaryShapExplainerNextSplit:
    """Tests for _get_next_split behavior."""

    @pytest.fixture
    def explainer(self) -> ComplementaryShapExplainer:
        """Provides a default explainer instance for testing."""
        return DummyComplementaryExplainer(num_samples=4)

    def test_complementary_mask_pair_generation(self) -> None:
        """Tests that complementary mask pairs are generated correctly."""
        explainer = DummyComplementaryExplainer(num_samples=8)
        explainer._initialize_state()
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
        explainer._initialize_state()
        explainer._first_call = False
        result = explainer._get_next_split(n=3, device=torch.device("cpu"), generated_masks_num=9)
        assert result is None

    def test_masks_generator_length_matches_budget(self) -> None:
        """__len__ on generator should reflect configured sampling budget."""
        explainer = DummyComplementaryExplainer(num_samples=6)
        explainer._initialize_state()
        mask_manager = MasksManager(chat=DummyChat())
        gen = explainer._get_masks_generator(
            mask_manager=mask_manager,
            device=torch.device("cpu"),
            masks=[],
        )
        assert len(gen) == explainer._get_num_splits(mask_manager.n)


class TestComplementaryShapExplainerCalculateShapValues:
    """Tests for _calculate_shap_values method."""

    def test_complementary_pairs_computation(self) -> None:
        """Tests that complementary pairs are computed correctly."""
        explainer = DummyComplementaryExplainer()
        explainer._initialize_state()
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
        assert torch.isfinite(result).all()

    def test_raises_for_non_complementary_pair(self) -> None:
        """Tests that ValueError is raised for non-complementary mask pairs."""
        explainer = DummyComplementaryExplainer()
        explainer._initialize_state()
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
        explainer._initialize_state()
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

    def test_raises_when_zero_mask_not_skipped(self) -> None:
        """SHAP calculation should fail if zero mask is present."""
        explainer = DummyComplementaryExplainer()
        explainer._initialize_state()
        explainer._zero_mask_skipped = False
        masks = torch.ones((2, 2), dtype=torch.bool)
        similarities = torch.ones(2)
        with pytest.raises(RuntimeError, match="Zero mask was not skipped"):
            _ = explainer._calculate_shap_values(masks, similarities, torch.device("cpu"))

    def test_raises_when_matrices_not_initialized(self) -> None:
        """Missing M or C matrices should raise."""
        explainer = DummyComplementaryExplainer()
        explainer._initialize_state()
        explainer._zero_mask_skipped = True
        masks = torch.ones((2, 2), dtype=torch.bool)
        similarities = torch.ones(2)
        with pytest.raises(RuntimeError, match="M and C matrices must be initialized"):
            _ = explainer._calculate_shap_values(masks, similarities, torch.device("cpu"))

    def test_result_matches_manual_computation(self) -> None:
        """Compare result against manual ratio computation for small example."""
        explainer = DummyComplementaryExplainer()
        explainer._initialize_state()
        device = torch.device("cpu")
        masks = torch.tensor(
            [
                [True, True],
                [True, False],
                [False, True],
            ],
            dtype=torch.bool,
        )
        explainer._M = torch.tensor(
            [
                [0.0, 2.0, 1.0],
                [0.0, 1.0, 2.0],
            ],
            device=device,
        )
        explainer._C = torch.tensor(
            [
                [0.0, 4.0, 3.0],
                [0.0, 2.0, 6.0],
            ],
            device=device,
        )
        explainer._zero_mask_skipped = True
        similarities = torch.tensor([1.0, 2.0, 3.0])

        # expect _calculate_C_matrix to update C for the complementary pairs without changing base columns
        result = explainer._calculate_shap_values(masks=masks, similarities=similarities, device=device)

        M = explainer._M[:, 1:]
        C = explainer._C[:, 1:]
        ratio = torch.zeros_like(C)
        non_zero = M != 0
        ratio[non_zero] = C[non_zero] / M[non_zero]
        expected = torch.sum(ratio, dim=1) / M.shape[0]

        torch.testing.assert_close(result, expected)

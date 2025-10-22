"""Tests for the McShapExplainer class in monte_carlo.py."""

import pytest
import torch
from mllm_shap.shap.monte_carlo import McShapExplainer


class TestMcShapExplainer:
    """Tests for the McShapExplainer class."""

    @staticmethod
    @pytest.fixture
    def setup_explainer() -> McShapExplainer:
        """Fixture that returns a default explainer for reuse."""
        return McShapExplainer(num_samples=5)

    def test_init_with_valid_num_samples(self) -> None:
        """Test initialization with valid num_samples."""
        explainer = McShapExplainer(num_samples=10)
        assert explainer.num_samples == 10
        assert explainer.fraction == 0.6

    def test_init_with_valid_fraction_only(self) -> None:
        """Test initialization with valid fraction only."""
        explainer = McShapExplainer(num_samples=None, fraction=0.8)
        assert explainer.fraction == 0.8
        assert explainer.num_samples is None

    def test_init_with_invalid_fraction_type(self) -> None:
        """Test initialization with invalid fraction type."""
        with pytest.raises(ValueError, match="fraction must be a float"):
            McShapExplainer(num_samples=None, fraction="0.5")

    def test_init_with_invalid_fraction_range(self):
        """Test initialization with invalid fraction range."""
        with pytest.raises(ValueError):
            McShapExplainer(num_samples=None, fraction=1.5)
        with pytest.raises(ValueError):
            McShapExplainer(num_samples=None, fraction=0.0)

    def test_init_with_invalid_num_samples_type(self) -> None:
        """Test initialization with invalid num_samples type."""
        with pytest.raises(ValueError, match="num_samples must be a positive integer."):
            McShapExplainer(num_samples="10")

    def test_init_with_invalid_num_samples_value(self) -> None:
        """Test initialization with invalid num_samples value."""
        with pytest.raises(ValueError, match="num_samples must be a positive integer."):
            McShapExplainer(num_samples=0)

    def test_init_with_both_none(self) -> None:
        """Test initialization with both num_samples and fraction as None."""
        with pytest.raises(ValueError, match="Either num_samples or fraction must be provided."):
            McShapExplainer(num_samples=None, fraction=None)

    def test_generate_masks_with_minimal_num_samples(self) -> None:
        """Test mask generation with minimal number of masks."""
        explainer = McShapExplainer(num_samples=-1)  # <- use -1
        n = 3
        masks = explainer._generate_masks(n=n, device=torch.device("cpu"))
        assert masks.shape == (n + 1, n)  # empty mask + single-feature masks
        assert masks.dtype == torch.bool
        assert any(mask.sum() == 0 for mask in masks)  # empty mask exists
        assert sum(mask.sum() == 1 for mask in masks) == n  # n single-feature masks

    def test_generate_masks_with_fraction(self) -> None:
        """Test mask generation with fraction parameter."""
        explainer = McShapExplainer(num_samples=None, fraction=0.5)
        n = 3
        torch.manual_seed(42)
        masks = explainer._generate_masks(n=n, device=torch.device("cpu"))
        total_possible = 2**n - 1
        expected_max = int(total_possible * 0.5)
        assert masks.shape[0] <= expected_max + n + 1

    def test_generate_masks_with_num_samples_too_small(self) -> None:
        """Test mask generation with num_samples less than number of features."""
        explainer = McShapExplainer(num_samples=2)
        with pytest.raises(ValueError, match="num_samples must be at least equal to the number of features."):
            explainer._generate_masks(n=3, device=torch.device("cpu"))

    def test_generate_masks_with_num_samples_exceeding_possible(self) -> None:
        """Test mask generation with num_samples exceeding total possible masks."""
        explainer = McShapExplainer(num_samples=100)
        n = 3
        masks = explainer._generate_masks(n=n, device=torch.device("cpu"))
        max_masks = 2**n - 1
        assert masks.shape[0] <= max_masks

    def test_generate_masks_respects_existing_masks(self) -> None:
        """Test that generated masks do not duplicate existing masks."""
        explainer = McShapExplainer(num_samples=5)
        n = 3
        existing_masks = torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=torch.bool)
        masks = explainer._generate_masks(n=n, device=torch.device("cpu"), existing_masks=existing_masks)
        existing = {tuple(row.tolist()) for row in existing_masks}
        generated = {tuple(row.tolist()) for row in masks}
        assert not (existing & generated)

    def test_calculate_shap_values_correctness(self, setup_explainer: McShapExplainer) -> None:
        """Test correctness of SHAP value calculation."""
        explainer = setup_explainer
        masks = torch.tensor(
            [
                [1, 0],
                [0, 1],
                [1, 1],
            ],
            dtype=torch.bool,
        )
        similarities = torch.tensor([0.2, 0.8, 0.6])
        device = torch.device("cpu")
        shap_values = explainer._calculate_shap_values(masks, similarities, device)
        assert shap_values.shape == (2,)
        assert torch.isfinite(shap_values).all()
        assert not torch.allclose(shap_values, torch.zeros_like(shap_values))

    def test_update_masks_if_not_present_adds_unique(self) -> None:
        """Test that unique masks are added."""
        mask = torch.tensor([1, 0, 1], dtype=torch.bool)
        masks = []
        existing = set()
        updated = McShapExplainer._McShapExplainer__update_masks_if_not_present(mask, masks, existing)
        assert len(updated) == 1
        assert tuple(mask.tolist()) in existing

    def test_update_masks_if_not_present_skips_duplicate(self) -> None:
        """Test that duplicate masks are not added."""
        mask = torch.tensor([1, 0, 1], dtype=torch.bool)
        masks = [mask]
        existing = {tuple(mask.tolist())}
        updated = McShapExplainer._McShapExplainer__update_masks_if_not_present(mask, masks, existing)
        assert len(updated) == 1

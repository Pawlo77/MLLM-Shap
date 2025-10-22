"""Tests for the PreciseShapExplainer class in precise.py."""

import pytest
import torch
from mllm_shap.shap.precise import generate_all_masks, PreciseShapExplainer


class TestGenerateAllMasks:
    """Tests for the generate_all_masks helper function."""

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_excludes_all_ones_and_has_correct_shape(self, n: int) -> None:
        """Test that all-ones mask is excluded and shape is correct."""
        device = torch.device("cpu")
        masks = generate_all_masks(n, device)
        # There should be (2^n - 1) masks total
        expected_rows = 2**n - 1
        assert masks.shape == (expected_rows, n)
        # Verify all-ones mask is excluded
        assert not any(mask.all() for mask in masks)
        # Verify boolean dtype
        assert masks.dtype == torch.bool

    def test_content_for_n2(self) -> None:
        """Test that the generated masks for n=2 are correct."""
        masks = generate_all_masks(2, torch.device("cpu"))
        expected = torch.tensor(
            [
                [0, 0],
                [0, 1],
                [1, 0],
            ],
            dtype=torch.bool,
        )
        assert torch.equal(masks, expected)


class TestPreciseShapExplainer:
    """Tests for the PreciseShapExplainer class."""

    def test_generate_masks_without_existing(self) -> None:
        """Test mask generation without existing masks."""
        explainer = PreciseShapExplainer()
        n = 3
        masks = explainer._generate_masks(n=n, device=torch.device("cpu"))
        # Should produce all masks except all-ones
        expected = generate_all_masks(n, torch.device("cpu"))
        assert torch.equal(masks, expected)

    def test_generate_masks_with_existing_masks(self) -> None:
        """Test mask generation with existing masks provided."""
        explainer = PreciseShapExplainer()
        n = 3
        all_masks = generate_all_masks(n, torch.device("cpu"))
        existing_masks = all_masks[:2]
        unseen_masks = explainer._generate_masks(
            n=n,
            device=torch.device("cpu"),
            existing_masks=existing_masks,
        )
        # Ensure unseen masks are the difference
        expected = torch.stack(
            [m for m in all_masks if tuple(m.tolist()) not in {tuple(x.tolist()) for x in existing_masks}]
        )
        assert torch.equal(unseen_masks, expected)

    def test_generate_masks_with_all_existing_masks_returns_empty(self) -> None:
        """Test that providing all possible masks returns an empty tensor."""
        explainer = PreciseShapExplainer()
        n = 2
        existing_masks = generate_all_masks(n, torch.device("cpu"))
        result = explainer._generate_masks(n=n, device=torch.device("cpu"), existing_masks=existing_masks)
        assert result.shape == (0, n)
        assert result.dtype == torch.bool

    def test_calculate_shap_values_correctness(self) -> None:
        """
        Test correctness of SHAP value calculation for a known function.

        Simple additive function f(x1, x2) = x1 + 2*x2
        SHAP values should approximately equal the feature contributions.
        """
        explainer = PreciseShapExplainer()
        device = torch.device("cpu")

        n = 2
        masks = generate_all_masks(n, device)
        masks = torch.cat([masks, torch.ones((1, n), dtype=torch.bool, device=device)], dim=0)  # add all-ones mask
        # Compute f(S): for each mask, sum of weights where feature=1
        weights = torch.tensor([1.0, 2.0], device=device)
        similarities = (masks.float() * weights).sum(dim=1)

        shap_values = explainer._calculate_shap_values(masks, similarities, device)

        # Expected SHAP values for linear additive functions are their coefficients
        torch.testing.assert_close(shap_values, weights, rtol=1e-5, atol=1e-5)

    def test_calculate_shap_values_handles_single_feature(self) -> None:
        """Test SHAP value calculation with a single feature."""
        explainer = PreciseShapExplainer()
        device = torch.device("cpu")

        masks = generate_all_masks(1, device)
        similarities = torch.tensor([0.5])
        shap_values = explainer._calculate_shap_values(masks, similarities, device)

        assert shap_values.shape == (1,)
        assert torch.isfinite(shap_values).all()

    def test_calculate_shap_values_device_consistency(self) -> None:
        """Test that SHAP values are computed on the correct device and dtype."""
        explainer = PreciseShapExplainer()
        device = torch.device("cpu")

        masks = generate_all_masks(2, device)
        similarities = torch.tensor([0.1, 0.5, 0.7], device=device)
        shap_values = explainer._calculate_shap_values(masks, similarities, device)

        assert shap_values.device == device
        assert shap_values.dtype == similarities.dtype

"""Tests for the PreciseShapExplainer class in precise.py."""

from itertools import product

import pytest
import torch
from mllm_shap.shap.precise import PreciseShapExplainer


class TestPreciseShapExplainer:
    """Tests for the PreciseShapExplainer class."""

    def test_get_num_splits_returns_correct_value(self) -> None:
        """Test that _get_num_splits returns 2**n - 1 for given target length."""
        explainer = PreciseShapExplainer()
        explainer._initialize_state()
        n = 3
        result = explainer._get_num_splits(n)
        assert result == 2**n - 1

    def test_get_next_split_generates_all_masks(self) -> None:
        """Test that _get_next_split generates all possible masks except all-ones."""
        explainer = PreciseShapExplainer()
        explainer._initialize_state()
        n = 3
        device = torch.device("cpu")

        generated = []
        mask = explainer._get_next_split(n, device, 0)
        generated.append(mask)
        i = 1
        while True:
            mask = explainer._get_next_split(n, device, i)
            if mask is None:
                break
            generated.append(mask)
            i += 1

        generated = torch.stack(generated)
        expected = torch.tensor(
            [split for split in product([0, 1], repeat=n) if sum(split) != n],
            dtype=torch.bool,
            device=device,
        )

        assert torch.equal(
            torch.sort(generated.float(), dim=0)[0],
            torch.sort(expected.float(), dim=0)[0],
        )

    def test_get_next_split_raises_if_generator_missing(self) -> None:
        """Test that calling _get_next_split without initialization raises an error."""
        explainer = PreciseShapExplainer()
        explainer._initialize_state()
        explainer._first_call = False
        with pytest.raises(RuntimeError, match="Splits generator is not present."):
            explainer._get_next_split(n=3, device=torch.device("cpu"), generated_masks_num=1)

    def test_get_next_split_returns_none_after_completion(self) -> None:
        """Test that _get_next_split returns None after all masks have been generated."""
        explainer = PreciseShapExplainer()
        explainer._initialize_state()
        n = 2
        device = torch.device("cpu")

        # Exhaust generator
        for i in range(explainer._get_num_splits(n) + 1):
            mask = explainer._get_next_split(n, device, i)
        assert mask is None

    def test_calculate_shap_values_correctness(self) -> None:
        """
        Test correctness of SHAP value calculation for a known additive function.

        Simple additive function f(x1, x2) = x1 + 2*x2.
        SHAP values should match the coefficients [1.0, 2.0].
        """
        explainer = PreciseShapExplainer()
        explainer._initialize_state()
        device = torch.device("cpu")
        n = 2

        # Generate all possible masks including all-ones for f(S)
        masks = torch.tensor(
            [split for split in product([0, 1], repeat=n)],
            dtype=torch.bool,
            device=device,
        )
        weights = torch.tensor([1.0, 2.0], device=device)
        similarities = (masks.float() * weights).sum(dim=1)

        shap_values = explainer._calculate_shap_values(masks, similarities, device)

        torch.testing.assert_close(shap_values, weights, rtol=1e-5, atol=1e-5)

    def test_calculate_shap_values_handles_single_feature(self) -> None:
        """Test SHAP value computation for single-feature case."""
        explainer = PreciseShapExplainer()
        explainer._initialize_state()
        device = torch.device("cpu")

        masks = torch.tensor([[False], [True]], dtype=torch.bool, device=device)
        similarities = torch.tensor([0.0, 1.0], device=device)
        shap_values = explainer._calculate_shap_values(masks, similarities, device)

        assert shap_values.shape == (1,)
        assert torch.isfinite(shap_values).all()

    def test_calculate_shap_values_device_consistency(self) -> None:
        """Test that SHAP values are computed on the same device and dtype as inputs."""
        explainer = PreciseShapExplainer()
        explainer._initialize_state()
        device = torch.device("cpu")

        masks = torch.tensor(
            [[False, False], [True, False], [False, True]],
            dtype=torch.bool,
            device=device,
        )
        similarities = torch.tensor([0.1, 0.5, 0.7], device=device)
        shap_values = explainer._calculate_shap_values(masks, similarities, device)

        assert shap_values.device == device
        assert shap_values.dtype == similarities.dtype

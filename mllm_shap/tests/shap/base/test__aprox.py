"""Unit tests for BaseShapApproximation class."""

import pytest
import torch
from mllm_shap.shap.base.approx import BaseShapApproximation
from torch import Tensor


class DummyExplainer(BaseShapApproximation):
    """Concrete subclass for testing abstract BaseShapApproximation."""

    def _get_next_split(self, target_length: int, device: torch.device, generated_masks: int) -> Tensor | None:
        return None

    def _get_num_splits(self, target_length: int) -> int | None:
        return None

    def _calculate_shap_values(
        self,
        masks: Tensor,
        similarities: Tensor,
        device: torch.device,
    ) -> Tensor:
        return torch.zeros((masks.size(0),), device=device)


class TestBaseShapApproximation:
    """Unit tests for BaseShapApproximation initialization and validation."""

    def test_init_with_valid_fraction(self):
        """Should correctly initialize with a valid fraction and no num_samples."""
        explainer = DummyExplainer(num_samples=None, fraction=0.8)
        assert explainer.num_samples is None
        assert explainer.fraction == 0.8

    def test_init_with_valid_num_samples(self):
        """Should correctly initialize with a valid num_samples and default fraction."""
        explainer = DummyExplainer(num_samples=10)
        assert explainer.num_samples == 10
        assert explainer.fraction == 0.6  # default

    def test_init_with_minimal_num_samples(self):
        """Should accept num_samples = -1 as valid (minimal sampling mode)."""
        explainer = DummyExplainer(num_samples=-1)
        assert explainer.num_samples == -1

    def test_init_raises_if_both_none(self):
        """Should raise ValueError if both num_samples and fraction are None."""
        with pytest.raises(ValueError, match="Either num_samples or fraction must be provided"):
            DummyExplainer(num_samples=None, fraction=None)

    def test_init_raises_for_invalid_fraction_type(self):
        """Should raise ValueError if fraction is not a float."""
        with pytest.raises(ValueError, match="fraction must be a float"):
            DummyExplainer(fraction="invalid")

    @pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
    def test_init_raises_for_invalid_fraction_value(self, fraction):
        """Should raise ValueError for fractions outside (0,1]."""
        with pytest.raises(ValueError, match="fraction must be a float in the range"):
            DummyExplainer(fraction=fraction)

    @pytest.mark.parametrize("num_samples", [0, -5, 2.5, "abc"])
    def test_init_raises_for_invalid_num_samples(self, num_samples):
        """Should raise ValueError for invalid num_samples values."""
        with pytest.raises(ValueError, match="num_samples must be a positive integer"):
            DummyExplainer(num_samples=num_samples)

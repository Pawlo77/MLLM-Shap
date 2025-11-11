"""Unit tests for BaseShapApproximation class."""

from typing import Any

import pytest
import torch
from mllm_shap.shap.base.approx import BaseShapApproximation
from torch import Tensor


class DummyExplainer(BaseShapApproximation):
    """Concrete subclass for testing abstract BaseShapApproximation."""

    def _get_next_split(
        self, n: int, device: torch.device, generated_masks: int, extra_arg: Any = None
    ) -> Tensor | None:
        return None

    def _get_num_splits(self, n: int) -> int:
        # Return a large enough value to avoid budget errors
        return 100

    def _calculate_shap_values(
        self,
        masks: Tensor,
        similarities: Tensor,
        device: torch.device,
    ) -> Tensor:
        return torch.zeros((masks.size(0),), device=device)


class TestBaseShapApproximationInit:
    """Unit tests for BaseShapApproximation initialization and validation."""

    def test_init_with_valid_fraction(self) -> None:
        """Should correctly initialize with a valid fraction and no num_samples."""
        explainer = DummyExplainer(num_samples=None, fraction=0.8)
        assert explainer.num_samples is None
        assert explainer.fraction == 0.8

    def test_init_with_valid_num_samples(self) -> None:
        """Should correctly initialize with a valid num_samples and default fraction."""
        explainer = DummyExplainer(num_samples=10)
        assert explainer.num_samples == 10
        assert explainer.fraction == 0.6  # default

    def test_init_with_minimal_num_samples(self) -> None:
        """Should accept num_samples = -1 as valid (minimal sampling mode)."""
        explainer = DummyExplainer(num_samples=-1)
        assert explainer.num_samples == -1

    def test_init_raises_if_both_none(self) -> None:
        """Should raise ValueError if both num_samples and fraction are None."""
        with pytest.raises(ValueError, match="Either num_samples or fraction must be provided"):
            DummyExplainer(num_samples=None, fraction=None)

    def test_init_raises_for_invalid_fraction_type(self) -> None:
        """Should raise ValueError if fraction is not a float."""
        with pytest.raises(ValueError, match="fraction must be a float"):
            DummyExplainer(fraction="invalid")

    @pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
    def test_init_raises_for_invalid_fraction_value(self, fraction: float) -> None:
        """Should raise ValueError for fractions outside (0,1]."""
        with pytest.raises(ValueError, match="fraction must be a float in the range"):
            DummyExplainer(fraction=fraction)

    @pytest.mark.parametrize("num_samples", [0, -5, 2.5, "abc"])
    def test_init_raises_for_invalid_num_samples(self, num_samples: Any) -> None:
        """Should raise ValueError for invalid num_samples values."""
        with pytest.raises(ValueError, match="num_samples must be a positive integer"):
            DummyExplainer(num_samples=num_samples)


class TestGenerateMinimalSplits:
    """Tests for static minimal mask generation."""

    def test_output_shape(self) -> None:
        """Test that generate_minimal_splits produces correct shape."""
        device = torch.device("cpu")
        n = 4
        masks = DummyExplainer._generate_minimal_splits(n, device)
        assert masks.shape == (n + 1, n)

    def test_first_row_is_all_false(self) -> None:
        """Test that the first row of generated minimal splits is all False."""
        device = torch.device("cpu")
        n = 3
        masks = DummyExplainer._generate_minimal_splits(n, device)
        assert torch.equal(masks[0], torch.zeros(n, dtype=torch.bool, device=device))

    def test_each_subsequent_row_has_single_false(self) -> None:
        """Each subsequent row should have exactly one False at the correct position."""
        device = torch.device("cpu")
        n = 5
        masks = DummyExplainer._generate_minimal_splits(n, device)
        for i in range(1, n + 1):
            row = masks[i]
            assert torch.sum(~row) == 1  # exactly one False
            false_index = torch.where(~row)[0].item()
            assert false_index == i - 1

    def test_dtype_and_device(self) -> None:
        """Generated minimal splits have correct dtype and device."""
        device = torch.device("cpu")
        n = 2
        masks = DummyExplainer._generate_minimal_splits(n, device)
        assert masks.dtype == torch.bool
        assert masks.device == device


class TestGetNextSplitBase:
    """Tests for _get_next_split_base behavior and internal logic."""

    def setup_method(self) -> None:
        self.device = torch.device("cpu")
        self.n = 3
        self.explainer = DummyExplainer(num_samples=5)
        # initialize internal state
        self.explainer._first_call = True
        self.explainer._zero_mask_skipped = False
        self.explainer._base_masks = None
        self.explainer._base_calls_num = 0

    def test_first_call_generates_base_masks(self) -> None:
        """Should generate base masks on first call."""
        mask = self.explainer._get_next_split_base(
            n=self.n,
            device=self.device,
            generated_masks_num=0,
        )
        assert isinstance(mask, Tensor)
        assert self.explainer._base_masks is not None
        assert mask.shape == (self.n,)

    def test_returns_none_when_generated_masks_exceed_base(self) -> None:
        """Should return None when generated masks exceed base mask count."""
        self.explainer._get_next_split_base(self.n, self.device, 0)
        num_base = self.explainer._base_masks.shape[0]
        result = self.explainer._get_next_split_base(
            n=self.n,
            device=self.device,
            generated_masks_num=num_base,
        )
        assert result is None

    def test_runtime_error_if_base_masks_missing(self) -> None:
        """Should raise RuntimeError if base masks are unexpectedly missing."""
        self.explainer.include_minimal_masks = True
        self.explainer._base_masks = None
        self.explainer._first_call = False
        with pytest.raises(RuntimeError, match="Base masks are not present"):
            self.explainer._get_next_split_base(
                n=self.n,
                device=self.device,
                generated_masks_num=1,
            )

    def test_runtime_error_multiple_base_rejected(self) -> None:
        """Should raise RuntimeError when multiple base masks are rejected."""
        self.explainer._first_call = False
        self.explainer._zero_mask_skipped = True
        self.explainer._base_masks = torch.zeros((2, 3), dtype=torch.bool)
        self.explainer._base_calls_num = 0
        with pytest.raises(RuntimeError, match="Multiple base masks were rejected"):
            self.explainer._get_next_split_base(
                n=self.n,
                device=self.device,
                generated_masks_num=0,
            )

"""Tests for SHAP normalizers."""

import pytest
import torch
from mllm_shap.shap.normalizers import (
    IdentityNormalizer,
    AbsSumNormalizer,
    PowerShiftNormalizer,
)


@pytest.fixture
def shap_values() -> torch.Tensor:
    """Fixture for sample SHAP values tensor."""
    return torch.tensor([1.0, -2.0, 3.0])


@pytest.fixture
def zero_shap_values() -> torch.Tensor:
    """Fixture for zero SHAP values tensor."""
    return torch.tensor([0.0, 0.0, 0.0])


class TestIdentityNormalizer:
    """Tests for IdentityNormalizer."""

    def test_returns_same_tensor(self, shap_values: torch.Tensor) -> None:
        """Test that the normalizer returns the same tensor."""
        normalizer = IdentityNormalizer()
        out = normalizer(shap_values)
        assert torch.equal(out, shap_values)


class TestAbsSumNormalizer:
    """Tests for AbsSumNormalizer."""

    def test_normal_case(self, shap_values: torch.Tensor) -> None:
        """Test normal case where sum of absolutes is non-zero."""
        normalizer = AbsSumNormalizer()
        out = normalizer(shap_values)
        expected = shap_values / shap_values.abs().sum()
        assert torch.allclose(out, expected)

    def test_zero_case(self, zero_shap_values: torch.Tensor) -> None:
        """Test case where sum of absolutes is zero."""
        normalizer = AbsSumNormalizer()
        out = normalizer(zero_shap_values)
        # Should return the same tensor if sum of absolutes is 0
        assert torch.equal(out, zero_shap_values)


class TestPowerShiftNormalizer:
    """Tests for PowerShiftNormalizer."""

    def test_default_power(self, shap_values: torch.Tensor) -> None:
        """Test normalization with default power of 1.0."""
        normalizer = PowerShiftNormalizer()
        out = normalizer(shap_values)
        # Output should sum to 1
        assert torch.isclose(out.sum(), torch.tensor(1.0))
        # All elements should be non-negative
        assert (out >= 0).all()

    def test_custom_power(self, shap_values: torch.Tensor) -> None:
        """Test normalization with custom power."""
        normalizer = PowerShiftNormalizer(power=2.0)
        out = normalizer(shap_values)
        assert torch.isclose(out.sum(), torch.tensor(1.0))
        assert (out >= 0).all()

    def test_zero_input(self, zero_shap_values: torch.Tensor) -> None:
        """Test case where input SHAP values are all zero."""
        normalizer = PowerShiftNormalizer()
        out = normalizer(zero_shap_values)
        # Should return original tensor if sum after power transform is 0
        assert torch.equal(out, zero_shap_values)

    def test_invalid_power(self) -> None:
        """Test that invalid power values raise ValueError."""
        with pytest.raises(ValueError):
            PowerShiftNormalizer(power=0)
        with pytest.raises(ValueError):
            PowerShiftNormalizer(power=-1)

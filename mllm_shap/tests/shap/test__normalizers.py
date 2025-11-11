"""Tests for SHAP normalizers."""

import pytest
import torch
from mllm_shap.shap.normalizers import (
    AbsSumNormalizer,
    IdentityNormalizer,
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

    def test_does_not_modify_input(self, shap_values: torch.Tensor) -> None:
        """Ensure the input tensor is not modified in-place."""
        normalizer = IdentityNormalizer()
        original = shap_values.clone()
        _ = normalizer(shap_values)
        assert torch.equal(shap_values, original)

    def test_handles_higher_dimensional_tensor(self) -> None:
        """Validate the normalizer supports non-1D tensors."""
        normalizer = IdentityNormalizer()
        values = torch.tensor([[1.0, -1.0], [0.5, -0.5]])
        out = normalizer(values)
        assert torch.equal(out, values)


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

    def test_abs_values_sum_to_one(self, shap_values: torch.Tensor) -> None:
        """The absolute values of the normalized tensor should sum to 1."""
        normalizer = AbsSumNormalizer()
        out = normalizer(shap_values)
        assert torch.isclose(out.abs().sum(), torch.tensor(1.0), atol=1e-6)

    def test_preserves_zero_entries(self) -> None:
        """Zero entries should remain zero after normalization."""
        normalizer = AbsSumNormalizer()
        values = torch.tensor([0.0, 2.0, -3.0, 0.0])
        out = normalizer(values)
        assert torch.isclose(out[0], torch.tensor(0.0))
        assert torch.isclose(out[3], torch.tensor(0.0))


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

    def test_fractional_power_preserves_order(self, shap_values: torch.Tensor) -> None:
        """Check that relative ordering is preserved after normalization."""
        normalizer = PowerShiftNormalizer(power=0.5)
        out = normalizer(shap_values)
        sorted_indices = shap_values.argsort()
        assert torch.equal(out.argsort(), sorted_indices)
        assert torch.isclose(out.sum(), torch.tensor(1.0), atol=1e-6)

    def test_all_equal_values_returns_original(self) -> None:
        """When all values become zero after shifting, return the input tensor."""
        normalizer = PowerShiftNormalizer()
        shap_values = torch.tensor([-1.0, -1.0, -1.0])
        out = normalizer(shap_values)
        assert torch.equal(out, shap_values)
        assert torch.isclose(out.sum(), torch.tensor(-3.0))

    def test_matches_expected_manual_computation(self) -> None:
        """Compare with manual power-shift implementation for transparency."""
        normalizer = PowerShiftNormalizer(power=2.0)
        shap_values = torch.tensor([-2.0, 1.0, 4.0])
        out = normalizer(shap_values)

        shifted = shap_values - shap_values.min()
        powered = shifted.pow(2.0)
        expected = powered / powered.sum()
        assert torch.allclose(out, expected)


class TestMinMaxNormalizer:
    """Tests for MinMaxNormalizer."""

    def test_normal_case(self, shap_values: torch.Tensor) -> None:
        """Test normal case where min != max."""
        from mllm_shap.shap.normalizers import MinMaxNormalizer

        normalizer = MinMaxNormalizer()
        out = normalizer(shap_values)

        # Should be between 0 and 1
        assert torch.all((out >= 0) & (out <= 1))
        # Should sum to 1
        assert torch.isclose(out.sum(), torch.tensor(1.0), atol=1e-6)

        # Manual expected computation
        min_val, max_val = shap_values.min(), shap_values.max()
        expected = (shap_values - min_val) / (max_val - min_val)
        expected /= expected.sum()
        assert torch.allclose(out, expected)

    def test_preserves_dtype(self) -> None:
        """The output tensor should keep the input dtype for downstream consistency."""
        from mllm_shap.shap.normalizers import MinMaxNormalizer

        normalizer = MinMaxNormalizer()
        shap_values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        out = normalizer(shap_values)
        assert out.dtype == torch.float64

    def test_two_element_tensor(self) -> None:
        """Verify behavior on the smallest non-trivial tensor size."""
        from mllm_shap.shap.normalizers import MinMaxNormalizer

        normalizer = MinMaxNormalizer()
        shap_values = torch.tensor([-1.0, 1.0])
        out = normalizer(shap_values)

        assert torch.all((out >= 0) & (out <= 1))
        assert torch.isclose(out.sum(), torch.tensor(1.0), atol=1e-6)

        min_val, max_val = shap_values.min(), shap_values.max()
        expected = (shap_values - min_val) / (max_val - min_val)
        expected /= expected.sum()
        assert torch.allclose(out, expected)

    def test_equal_values(self) -> None:
        """Test case where all SHAP values are identical (min == max)."""
        from mllm_shap.shap.normalizers import MinMaxNormalizer

        normalizer = MinMaxNormalizer()
        shap_values = torch.tensor([3.0, 3.0, 3.0])
        out = normalizer(shap_values)

        # Should return uniform distribution
        expected = torch.ones_like(shap_values) / len(shap_values)
        assert torch.allclose(out, expected)
        # Should sum to 1
        assert torch.isclose(out.sum(), torch.tensor(1.0), atol=1e-6)

    def test_zero_values(self, zero_shap_values: torch.Tensor) -> None:
        """Test case where all SHAP values are zeros."""
        from mllm_shap.shap.normalizers import MinMaxNormalizer

        normalizer = MinMaxNormalizer()
        out = normalizer(zero_shap_values)

        # Should return uniform distribution
        expected = torch.ones_like(zero_shap_values) / len(zero_shap_values)
        assert torch.allclose(out, expected)
        # Should sum to 1
        assert torch.isclose(out.sum(), torch.tensor(1.0), atol=1e-6)

    def test_negative_values(self) -> None:
        """Test normalization with negative values included."""
        from mllm_shap.shap.normalizers import MinMaxNormalizer

        shap_values = torch.tensor([-5.0, 0.0, 5.0])
        normalizer = MinMaxNormalizer()
        out = normalizer(shap_values)

        # Should be between 0 and 1 and sum to 1
        assert torch.all((out >= 0) & (out <= 1))
        assert torch.isclose(out.sum(), torch.tensor(1.0), atol=1e-6)

        # Manual expected computation
        min_val, max_val = shap_values.min(), shap_values.max()
        expected = (shap_values - min_val) / (max_val - min_val)
        expected /= expected.sum()
        assert torch.allclose(out, expected)

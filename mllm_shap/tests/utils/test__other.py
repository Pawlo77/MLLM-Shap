"""Tests for general utility functions."""

import pytest
import torch
from mllm_shap.utils.other import (
    extend_tensor,
    make_consecutive_ids_ignore_zero,
    raise_connector_error,
    safe_mask,
    safe_mask_unsqueeze,
)


@pytest.fixture
def tensor() -> torch.Tensor:
    """Fixture for sample tensor."""
    return torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


class TestRaiseConnectorError:
    """Tests for the raise_connector_error wrapper."""

    def test_successful_callable(self) -> None:
        """Callable executes correctly, returns expected result."""

        def add(a, b):
            return a + b

        result = raise_connector_error(add, 2, 3)
        assert result == 5

    def test_exception_wrapping(self) -> None:
        """Callable raises error, should be wrapped in RuntimeError."""

        def fail_fn():
            raise ValueError("Original error")

        with pytest.raises(RuntimeError) as exc_info:
            raise_connector_error(fail_fn)
        assert "Error occurred in connector implementation." in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, ValueError)


class TestSafeMask:
    """Tests for safe_mask function."""

    def test_regular_mask(self, tensor: torch.Tensor) -> None:
        mask = torch.tensor([True, False, True])
        masked = safe_mask(tensor, mask)
        expected = torch.tensor([[1.0, 3.0], [4.0, 6.0]])
        assert torch.equal(masked, expected)

    def test_empty_mask(self, tensor: torch.Tensor) -> None:
        mask = torch.tensor([False, False, False])
        masked = safe_mask(tensor, mask)
        assert masked.shape == (2, 0)  # maintains batch dimension
        assert masked.numel() == 0

    def test_single_dim_tensor(self) -> None:
        tensor = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.tensor([True, False, True])
        masked = safe_mask(tensor, mask)
        expected = torch.tensor([1.0, 3.0])
        assert torch.equal(masked, expected)


class TestSafeMaskUnsqueeze:
    """Tests for safe_mask_unsqueeze function."""

    def test_regular_mask(self, tensor: torch.Tensor) -> None:
        mask = torch.tensor([True, False, True])
        masked = safe_mask_unsqueeze(tensor, mask)
        expected = torch.tensor([[1.0, 3.0]])
        assert masked.shape == (1, 2)  # batch dimension maintained
        assert torch.equal(masked, expected)

    def test_empty_mask(self, tensor: torch.Tensor) -> None:
        mask = torch.tensor([False, False, False])
        masked = safe_mask_unsqueeze(tensor, mask)
        expected_shape = (tensor.shape[0], 0)
        assert masked.shape == expected_shape
        assert masked.numel() == 0


class TestMakeConsecutiveIdsIgnoreZero:
    """Tests for make_consecutive_ids_ignore_zero function."""

    def test_basic_case(self) -> None:
        """Converts non-zero IDs to consecutive integers, keeping zeros unchanged."""
        t = torch.tensor([0, 1, 1, 2, 2, 0, 4, 4])
        expected = torch.tensor([0, 1, 1, 2, 2, 0, 3, 3])
        result = make_consecutive_ids_ignore_zero(t)
        assert torch.equal(result, expected)

    def test_all_zeros(self) -> None:
        """Handles tensor with only zeros."""
        t = torch.zeros(5, dtype=torch.long)
        result = make_consecutive_ids_ignore_zero(t)
        assert torch.equal(result, t)

    def test_consecutive_groups(self) -> None:
        """Handles multiple non-zero groups in order of appearance."""
        t = torch.tensor([1, 1, 3, 3, 5, 5])
        expected = torch.tensor([1, 1, 2, 2, 3, 3])
        result = make_consecutive_ids_ignore_zero(t)
        assert torch.equal(result, expected)

    def test_with_repeated_ids_non_consecutive(self) -> None:
        """Ensures IDs that reappear later are remapped consistently."""
        t = torch.tensor([0, 1, 1, 0, 2, 6, 0, 0])
        expected = torch.tensor([0, 1, 1, 0, 2, 3, 0, 0])
        result = make_consecutive_ids_ignore_zero(t)
        assert torch.equal(result, expected)

    def test_device_support(self) -> None:
        """Works correctly on GPU (or MPS) devices if available."""
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            pytest.skip("No GPU/MPS device available.")

        t = torch.tensor([0, 2, 2, 5, 5, 0, 9, 9], device=device)
        expected = torch.tensor([0, 1, 1, 2, 2, 0, 3, 3], device=device)
        result = make_consecutive_ids_ignore_zero(t)
        assert torch.equal(result, expected)
        assert result.device == t.device


class TestExtendTensor:
    """Tests for the extend_tensor function."""

    def test_extend_short_tensor(self) -> None:
        """Extends a tensor to the target length using the specified fill value."""
        t = torch.tensor([1, 2, 3])
        result = extend_tensor(t, target_length=6, fill_value=0)
        expected = torch.tensor([1, 2, 3, 0, 0, 0])
        assert torch.equal(result, expected)

    def test_no_extension_needed(self) -> None:
        """Returns the tensor unchanged if it already meets or exceeds target length."""
        t = torch.tensor([1, 2, 3, 4])
        result = extend_tensor(t, target_length=3, fill_value=-1)
        assert torch.equal(result, t)

    def test_extend_with_negative_fill(self) -> None:
        """Supports custom fill values such as negative numbers."""
        t = torch.tensor([5, 6])
        result = extend_tensor(t, target_length=5, fill_value=-1)
        expected = torch.tensor([5, 6, -1, -1, -1])
        assert torch.equal(result, expected)

    def test_preserves_dtype_and_device(self) -> None:
        """Preserves tensor dtype and device."""
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        t = torch.tensor([1.5, 2.5], dtype=torch.float32, device=device)
        result = extend_tensor(t, target_length=4, fill_value=0.5)

        assert result.dtype == torch.float32
        assert result.device == t.device
        expected = torch.tensor(
            [1.5, 2.5, 0.5, 0.5], dtype=torch.float32, device=device
        )
        assert torch.equal(result, expected)

    def test_zero_length_tensor(self) -> None:
        """Handles extension of an empty tensor."""
        t = torch.tensor([], dtype=torch.int64)
        result = extend_tensor(t, target_length=3, fill_value=7)
        expected = torch.tensor([7, 7, 7], dtype=torch.int64)
        assert torch.equal(result, expected)

    @pytest.mark.parametrize(
        "target_length,fill_value,expected",
        [
            (5, 9, [1, 2, 9, 9, 9]),
            (2, 9, [1, 2]),
            (3, 0, [1, 2, 0]),
        ],
    )
    def test_multiple_cases(
        self, target_length: int, fill_value: int, expected: list[int]
    ) -> None:
        """Parameterized test for various input configurations."""
        t = torch.tensor([1, 2])
        result = extend_tensor(t, target_length=target_length, fill_value=fill_value)
        expected_tensor = torch.tensor(expected)
        assert torch.equal(result, expected_tensor)

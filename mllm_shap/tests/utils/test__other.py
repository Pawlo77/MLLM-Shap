"""Tests for general utility functions."""

import pytest
import torch
from mllm_shap.utils.other import raise_connector_error, safe_mask, safe_mask_unsqueeze


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

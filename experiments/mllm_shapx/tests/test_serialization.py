"""Tests for serialization module."""

import numpy as np
import torch

from ..src.serialization import (
    _safe_primitive,
)


class TestSafePrimitive:
    def test_bytes(self) -> None:
        result = _safe_primitive(b"hello")
        assert result == {"_binary": True, "num_bytes": 5}

    def test_numpy_scalar(self) -> None:
        result = _safe_primitive(np.float32(3.14))
        assert isinstance(result, float)
        assert abs(result - 3.14) < 0.01

    def test_torch_tensor(self) -> None:
        result = _safe_primitive(torch.tensor([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_nan_becomes_none(self) -> None:
        result = _safe_primitive(float("nan"))
        assert result is None

    def test_inf_becomes_none(self) -> None:
        result = _safe_primitive(float("inf"))
        assert result is None

    def test_regular_value_passthrough(self) -> None:
        assert _safe_primitive(42) == 42
        assert _safe_primitive("hello") == "hello"
        assert _safe_primitive(None) is None

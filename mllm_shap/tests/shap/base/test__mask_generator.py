"""Tests for shap.base._mask_generator API contract."""

import pytest
import torch

from mllm_shap.shap.base._mask_generator import MaskGenerator


class DummyMaskGenerator(MaskGenerator):
    def _mask_iter(self):
        yield torch.tensor([True, False], dtype=torch.bool), 123


class RaiseMaskGenerator(MaskGenerator):
    def _mask_iter(self):
        raise RuntimeError("boom")
        yield torch.tensor([True], dtype=torch.bool), 1


def test_mask_generator_iter_and_next() -> None:
    """Checks that mask generator iter and next."""
    gen = DummyMaskGenerator()

    assert iter(gen) is gen

    mask, mask_hash = next(gen)
    assert mask_hash == 123
    assert torch.equal(mask, torch.tensor([True, False], dtype=torch.bool))

    with pytest.raises(StopIteration):
        next(gen)


def test_mask_generator_send_none() -> None:
    """Checks that mask generator send none."""
    gen = DummyMaskGenerator()
    mask, mask_hash = gen.send(None)

    assert mask_hash == 123
    assert mask.dtype == torch.bool


def test_mask_generator_throw_propagates_exception() -> None:
    """Checks that mask generator throw propagates exception."""
    gen = DummyMaskGenerator()

    with pytest.raises(ValueError, match="x"):
        gen.throw(ValueError("x"))


def test_mask_generator_raises_from_iter() -> None:
    """Checks that mask generator raises from iter."""
    gen = RaiseMaskGenerator()

    with pytest.raises(RuntimeError, match="boom"):
        next(gen)

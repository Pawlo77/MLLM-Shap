"""Tests for mask encoding/decoding helpers."""

import torch

from mllm_shap.shap.masks import MaskCodec, MaskDedupIndex


def test_pack_unpack_roundtrip() -> None:
    mask = torch.tensor([True, False, True, True, False, False, True], dtype=torch.bool)

    packed = MaskCodec.pack(mask)
    restored = MaskCodec.unpack(packed)

    assert torch.equal(restored, mask)


def test_hash_stable_across_shapes() -> None:
    row = torch.tensor([[True, False, True, False]], dtype=torch.bool)
    flat = torch.tensor([True, False, True, False], dtype=torch.bool)

    assert MaskCodec.hash(row) == MaskCodec.hash(flat)


def test_dedup_index_tracks_uniques() -> None:
    index = MaskDedupIndex()

    assert index.add(11) is True
    assert index.add(11) is False
    assert index.add(12) is True
    assert index.contains(11) is True
    assert index.contains(99) is False
    assert len(index) == 2

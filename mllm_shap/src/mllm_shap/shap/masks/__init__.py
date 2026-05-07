"""Mask encoding and deduplication primitives."""

from .dedup_index import MaskDedupIndex
from .mask_codec import MaskCodec, PackedMask
from .mask_space import MaskSpace

__all__ = [
    "MaskCodec",
    "PackedMask",
    "MaskDedupIndex",
    "MaskSpace",
]

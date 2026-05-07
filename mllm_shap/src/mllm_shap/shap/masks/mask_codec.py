"""Bitset-oriented mask encoding and hashing utilities."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class PackedMask:
    """Packed bit representation of one boolean mask."""

    words: bytes
    n_bits: int


class MaskCodec:
    """Encode/decode boolean masks to packed bytes."""

    @staticmethod
    def normalize(mask: Tensor) -> Tensor:
        """Normalize incoming mask to 1D bool tensor."""
        if mask.ndim > 1:
            if mask.shape[0] != 1:
                raise ValueError("Mask must be 1D or have exactly one row.")
            mask = mask.squeeze(0)
        return mask.contiguous().bool()

    @staticmethod
    def pack(mask: Tensor) -> PackedMask:
        """Pack mask into little-endian bytes."""
        normalized = MaskCodec.normalize(mask)
        n_bits = int(normalized.numel())
        cpu_bits = normalized.to(device="cpu").numpy().astype(np.uint8)
        packed = np.packbits(cpu_bits, bitorder="little")
        return PackedMask(words=packed.tobytes(), n_bits=n_bits)

    @staticmethod
    def unpack(packed: PackedMask, device: torch.device | None = None) -> Tensor:
        """Unpack bytes back to 1D bool tensor."""
        raw = np.frombuffer(packed.words, dtype=np.uint8)
        bits = np.unpackbits(raw, bitorder="little")[: packed.n_bits].copy()
        return torch.from_numpy(bits.astype(np.bool_)).to(device=device)

    @staticmethod
    def hash(mask: Tensor) -> int:
        """Stable hash over packed mask bytes."""
        packed = MaskCodec.pack(mask)
        return hash((packed.words, packed.n_bits))

"""Mask deduplication indexes."""

from dataclasses import dataclass, field


@dataclass
class MaskDedupIndex:
    """Exact deduplication index for mask hashes."""

    _seen: set[int] = field(default_factory=set)
    """Set of previously observed mask hashes."""

    def contains(self, mask_hash: int) -> bool:
        """Check hash membership."""
        return mask_hash in self._seen

    def add(self, mask_hash: int) -> bool:
        """Insert hash, return True if it was new."""
        is_new = mask_hash not in self._seen
        if is_new:
            self._seen.add(mask_hash)
        return is_new

    def __len__(self) -> int:
        """Return number of unique hashes."""
        return len(self._seen)

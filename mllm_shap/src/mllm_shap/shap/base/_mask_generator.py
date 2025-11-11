"""Mask Generator API."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generator

from torch import Tensor


class MaskGenerator(Generator[tuple[Tensor | None, int], None, None], ABC):
    """Generator for producing unique masks for SHAP explainability."""

    generated_masks: int

    def __init__(self) -> None:
        """Initialize the MaskGenerator."""
        super().__init__()
        self.generated_masks = 0
        self._iter = self._mask_iter()

    def send(self, *args: Any, **kwargs: Any) -> tuple[Tensor | None, int]:
        return self._iter.send(*args, **kwargs)

    def throw(self, *args: Any, **kwargs: Any) -> tuple[Tensor | None, int]:
        return self._iter.throw(*args, **kwargs)

    @abstractmethod
    def _mask_iter(self) -> Generator[tuple[Tensor | None, int], None, None]:
        """Iterator that yields unique masks and their hashes."""

    def __iter__(self) -> "MaskGenerator":
        return self

    def __next__(self) -> tuple[Tensor | None, int]:
        return next(self._iter)

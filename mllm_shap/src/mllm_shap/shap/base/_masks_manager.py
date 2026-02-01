"""Mask manager for SHAP explainability."""

from __future__ import annotations

from abc import ABC, abstractmethod
from logging import Logger
from typing import Any, Generator

import torch
from torch import Tensor

from ...connectors.base.chat import BaseMllmChat
from ...utils.logger import get_logger

logger: Logger = get_logger(__name__)


# pylint: disable=duplicate-code
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


class NoTokensToExplainError(Exception):
    """Raised when there are no tokens to explain in the chat."""


class MasksManager:
    """Manages the generation and tracking of masks for SHAP explainability."""

    shap_values_mask: Tensor
    """1D boolean tensor indicating which positions to split."""

    n: int
    """Number of features to explain."""

    target_length: int
    """Length of the final masks to be generated."""

    _seen_masks: set[int]
    """Set of seen mask hashes to avoid duplicates."""

    def __init__(self, chat: BaseMllmChat, log_stats: bool = False) -> None:
        """
        Initialize the MasksManager.

        Args:
            chat: The chat object containing the mask and token information.
            log_stats: Whether to log statistics about the mask generation.
        Raises:
            NoTokensToExplainError: If there are no tokens to explain in the provided chat.
        """
        mask = chat.shap_values_mask
        if not mask.any():
            raise NoTokensToExplainError("There are no tokens to explain in the provided chat.")
        self.shap_values_mask = mask

        self.target_length = chat.input_tokens_num
        logger.debug("Generating masks for target length %d using provided mask.", self.target_length)

        n = int(mask.sum().item())
        if n == 0:
            raise NoTokensToExplainError("Mask must have at least one True value.")
        self.n = n

        self._seen_masks = set()

        if log_stats:
            logger.info(
                "Number of tokens for explainability: %d (up to %d additional calls)",
                self.n,
                self.max_masks_number,
            )

    @property
    def max_masks_number(self) -> int:
        """Maximum number of unique masks possible for n features."""
        return int(2**self.n - 1)

    def mark_seen(self, mask: Tensor | None = None, mask_hash: int | None = None) -> None:
        """
        Mark the provided mask as seen.

        Args:
            mask: 1D boolean tensor representing the mask to mark as seen.
            mask_hash: Hash of the mask to mark as seen.
        """
        mask_hash = MasksManager.__get_mask_hash(mask=mask, mask_hash=mask_hash)
        if mask_hash not in self._seen_masks:
            self._seen_masks.add(mask_hash)

    def seen(self, mask: Tensor | None = None, mask_hash: int | None = None) -> bool:
        """
        Check if the provided mask has been seen.

        Args:
            mask: 1D boolean tensor representing the mask to check.
            mask_hash: Hash of the mask to check.
        Returns:
            True if the mask has been seen, False otherwise.
        """
        mask_hash = MasksManager.__get_mask_hash(mask=mask, mask_hash=mask_hash)
        return mask_hash in self._seen_masks

    def get_initial_mask(self, device: torch.device) -> Tensor:
        """
        Get the initial masks: all-ones mask.

        Args:
            device: Device to create the mask on.
        Returns:
            Tensor of shape [1, n], dtype=torch.bool, representing the starting mask.
        """
        mask = self.prepare_mask(
            split=torch.ones((1, self.n), dtype=torch.bool, device=device),
            device=device,
        )
        if mask is None:
            raise ValueError("Starting mask cannot be None.")
        self.mark_seen(mask)
        return mask

    def prepare_mask(self, split: Tensor, device: torch.device) -> Tensor | None:
        """
        Prepare the mask by setting masked positions according to split
        and keeping unmasked positions always True.

        Args:
            split: Tensor of shape [1, num_masked], dtype=torch.bool representing the split mask.
            device: The device to create the masks on
        Returns:
            Tensor of shape [target_length, ], dtype=torch.bool representing the final mask,
                or None if the final mask has no True values.
        """
        prepared_mask = torch.zeros((self.target_length,), dtype=torch.bool, device=device)

        # Set masked positions according to splits
        prepared_mask[self.shap_values_mask] = split
        # Keep unmasked positions always True
        prepared_mask[~self.shap_values_mask] = True

        # Filter out rows that have no True values (completely empty masks)
        # it is a case scenario when all tokens are taken into account for splitting
        if not prepared_mask.any():
            return None
        return prepared_mask

    @staticmethod
    def get_hash(mask: Tensor) -> int:
        """
        Get the hash of the provided mask.

        Args:
            mask: 1D boolean tensor representing the mask.
        Returns:
            Hash of the mask.
        """
        if len(mask.shape) > 1:
            if mask.shape[0] != 1:
                raise ValueError("Mask must be a 1D tensor or a 2D tensor with a single row.")
            mask = mask.squeeze(0)
        return hash(tuple(mask.tolist()))

    @staticmethod
    def __get_mask_hash(mask: Tensor | None = None, mask_hash: int | None = None) -> int:
        """
        Get the hash of the provided mask.

        Args:
            mask: 1D boolean tensor representing the mask.
            mask_hash: Precomputed hash of the mask.
        Returns:
            Hash of the mask.
        Raises:
            ValueError: If neither mask nor mask_hash is provided.
        """
        if mask_hash is None:
            if mask is None:
                raise ValueError("Either mask or mask_hash must be provided.")
            mask_hash = MasksManager.get_hash(mask)
        return mask_hash

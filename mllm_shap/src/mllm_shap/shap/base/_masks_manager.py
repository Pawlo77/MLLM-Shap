"""Mask manager for SHAP explainability."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import Logger
from typing import TYPE_CHECKING, Any, Generator

import torch
from torch import Tensor

from ...connectors.base.chat import BaseMllmChat
from ...errors import MaskError
from ...utils.logger import get_logger
from ..masks import MaskCodec, MaskDedupIndex, MaskSpace

if TYPE_CHECKING:
    from ..core.telemetry import TelemetryProbe

logger: Logger = get_logger(__name__)


class NoTokensToExplainError(Exception):
    """Raised when chat has no explainable tokens."""


class MaskGenerator(ABC):
    """Compatibility iterator wrapper for mask generation strategies."""

    def __init__(self) -> None:
        self._iterator = self._mask_iter()

    @abstractmethod
    def _mask_iter(self) -> Generator[tuple[Tensor, int], Any, None]:
        """Return the internal generator producing `(mask, count)` tuples."""

    def __iter__(self) -> "MaskGenerator":
        return self

    def __next__(self) -> tuple[Tensor, int]:
        return next(self._iterator)

    def send(self, value: Any) -> tuple[Tensor, int]:
        return self._iterator.send(value)

    def throw(self, exc: BaseException) -> tuple[Tensor, int]:
        return self._iterator.throw(exc)


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

    @dataclass(frozen=True)
    class _MaskHashStrategy:
        """Stateless strategy for mask hashing."""

        @staticmethod
        def normalize(mask: Tensor) -> Tensor:
            """Normalize mask to 1D shape.

            Args:
                mask: Raw mask tensor.

            Returns:
                1D boolean mask tensor.

            Raises:
                MaskError: If mask has invalid shape.
            """
            if len(mask.shape) > 1:
                if mask.shape[0] != 1:
                    raise MaskError(
                        "Mask must be a 1D tensor or a 2D tensor with a single row."
                    )
                return mask.squeeze(0)
            return mask

        @staticmethod
        def hash(mask: Tensor) -> int:
            """Hash mask bytes using bit-packing to minimize GPU→CPU transfer.

            Args:
                mask: Mask tensor to hash.

            Returns:
                Integer hash value.
            """
            normalized = MasksManager._MaskHashStrategy.normalize(mask)
            return MaskCodec.hash(normalized)

    def __init__(
        self,
        chat: BaseMllmChat,
        log_stats: bool = False,
        probe: "TelemetryProbe | None" = None,
    ) -> None:
        """
        Initialize the MasksManager.

        Args:
            chat: The chat object containing the mask and token information.
            log_stats: Whether to log statistics about the mask generation.
            probe: Optional TelemetryProbe for metrics collection.
        Raises:
            NoTokensToExplainError: If there are no tokens to explain in the provided chat.
        """
        mask = chat.shap_values_mask
        if not mask.any():
            raise NoTokensToExplainError(
                "There are no tokens to explain in the provided chat."
            )
        self.shap_values_mask = mask
        self._mask_space = MaskSpace(
            shap_values_mask=mask,
            target_length=chat.input_tokens_num,
        )

        self.target_length = chat.input_tokens_num
        logger.debug(
            "Generating masks for target length %d using provided mask.",
            self.target_length,
        )

        n = int(mask.sum().item())
        if n == 0:
            raise NoTokensToExplainError("Mask must have at least one True value.")
        self.n = n

        self._seen_masks = set()
        self._dedup_index = MaskDedupIndex()
        self._probe = probe

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

    def mark_seen(
        self, mask: Tensor | None = None, mask_hash: int | None = None
    ) -> None:
        """
        Mark the provided mask as seen.

        Args:
            mask: 1D boolean tensor representing the mask to mark as seen.
            mask_hash: Hash of the mask to mark as seen.
        """
        mask_hash = MasksManager.__get_mask_hash(mask=mask, mask_hash=mask_hash)
        is_unique = self._dedup_index.add(mask_hash)
        if is_unique:
            self._seen_masks.add(mask_hash)
        # Record in telemetry whether this is a unique or duplicate mask
        if self._probe:
            self._probe.mask_generated(is_unique=is_unique, is_invalid=False)

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
        return self._dedup_index.contains(mask_hash)

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
        # Keep behavior resilient when tests/runtime mutate shap_values_mask directly.
        mask_space = MaskSpace(
            shap_values_mask=self.shap_values_mask,
            target_length=self.target_length,
        )
        prepared_mask = mask_space.materialize(split=split, device=device)

        # Filter out rows that have no True values (completely empty masks)
        # it is a case scenario when all tokens are taken into account for splitting
        if not prepared_mask.any():
            # Track invalid mask generation
            if self._probe:
                self._probe.mask_generated(is_unique=False, is_invalid=True)
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
        return MasksManager._MaskHashStrategy.hash(mask)

    @staticmethod
    def __get_mask_hash(
        mask: Tensor | None = None, mask_hash: int | None = None
    ) -> int:
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
                raise MaskError("Either mask or mask_hash must be provided.")
            mask_hash = MasksManager.get_hash(mask)
        return mask_hash

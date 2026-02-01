"""Cache manager for SHAP explainers."""

from logging import Logger

from torch import Tensor

from ...utils.logger import get_logger
from ._masks_manager import MasksManager

from ...connectors.base.chat import BaseMllmChat
from ...connectors.base.model_response import ModelResponse
from ...connectors.base.explainer_cache import ExplainerCache


logger: Logger = get_logger(__name__)


class CacheManager:
    """Cache manager for SHAP explainers."""

    extracted_num: int = 0
    """Number of masks extracted from the cache."""

    cache: ExplainerCache | None
    """The cache instance the cache manager is for."""

    _masks_manager: MasksManager
    """Masks manager instance."""

    _responses_map: dict[int, int] = {}
    """Map of mask hashes to model responses."""

    def __init__(self, chat: BaseMllmChat, explainer_hash: int) -> None:
        """
        Initialize CacheManager by extracting SHAP explainer cache from the full chat.

        Args:
            chat: The chat instance to manage the cache for.
            explainer_hash: The hash of the explainer instance.
        Raises:
            ValueError: If existing cache is invalid.
        """
        logger.debug("Getting or setting SHAP explainer cache for chat %s.", chat)

        self._masks_manager = MasksManager(chat=chat)

        cache = chat.cache
        if cache is not None:
            if cache.calculated_by != explainer_hash:
                raise ValueError("Existing SHAP cache was calculated by a different explainer instance.")
            if cache.chat != chat:
                raise ValueError("Existing SHAP cache is associated with a different chat instance.")
            if cache.had_different_masks:
                logger.warning(
                    "Existing SHAP cache for chat was calculated with external mask, no retrieval will be done.",
                )

            # Extend existing masks to match new masks size
            cache.extend_masks()

            for i, mask in enumerate(cache.masks):
                mask_hash = self._masks_manager.get_hash(mask)
                self._masks_manager.mark_seen(mask_hash=mask_hash)
                self._responses_map[mask_hash] = i

            logger.debug("Found existing SHAP cache for chat %s.", chat)

        self.cache = cache
        chat.cache = None

    def contains(self, mask: Tensor | None = None, mask_hash: int | None = None) -> bool:
        """
        Check if the provided mask is present in cache.

        Args:
            mask: 1D boolean tensor representing the mask to check.
            mask_hash: Hash of the mask to check.
        Returns:
            True if the mask is present in cache, False otherwise.
        """
        return self._masks_manager.seen(mask=mask, mask_hash=mask_hash)

    def extract(self, mask: Tensor | None = None, mask_hash: int | None = None) -> ModelResponse:
        """
        Extract the model response for the provided mask from cache.

        Args:
            mask: 1D boolean tensor representing the mask to extract.
            mask_hash: Hash of the mask to extract.
        Returns:
            The model response corresponding to the mask.
        Raises:
            ValueError: If no cache is associated with this CacheManager.
            KeyError: If the mask is not present in cache.
        """
        if self.cache is None:
            raise ValueError("No cache is associated with this CacheManager.")
        if mask_hash is None:
            if mask is None:
                raise ValueError("Either mask or mask_hash must be provided.")
            mask_hash = self._masks_manager.get_hash(mask)

        response_idx = self._responses_map.get(mask_hash)
        if response_idx is None:
            raise KeyError("Mask not found in cache.")

        self.extracted_num += 1
        return self.cache.responses[response_idx]

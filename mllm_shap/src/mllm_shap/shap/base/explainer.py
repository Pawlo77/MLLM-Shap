"""Base class for SHAP-based explanations."""

import gc
from abc import ABC, abstractmethod
from logging import Logger
from time import time
from typing import Any, cast

import torch
from torch import Tensor
from tqdm.auto import tqdm

from ...connectors.base.chat import AllTextTokensFilteredOutError, BaseMllmChat
from ...connectors.base.explainer_cache import ExplainerCache
from ...connectors.base.model import BaseMllmModel
from ...utils.logger import get_logger
from ..embeddings import MeanReducer
from ..enums import Mode
from ..normalizers import PowerShiftNormalizer
from ..similarity import CosineSimilarity
from ._validators import BaseShapCallConfig, BaseShapConfig
from .embeddings import BaseEmbeddingReducer, BaseExternalEmbedding
from .normalizers import BaseNormalizer
from .similarity import BaseEmbeddingSimilarity

logger: Logger = get_logger(__name__)


class NoTokensToExplainError(Exception):
    """Raised when there are no tokens to explain in the chat."""


class NotEnoughTokensToExplainError(Exception):
    """Raised when there are not enough tokens to explain in the chat."""


# pylint: disable=too-few-public-methods
class BaseShapExplainer(ABC):
    """Base class for SHAP-based explanations."""

    mode: Mode
    """The SHAP mode, either `STATIC` or `CONTEXTUAL`. Used if no :attr:`embedding_model` is provided."""

    embedding_model: BaseExternalEmbedding | None
    """The external embedding model to use. If provided, overrides :attr:`mode`."""

    embedding_reducer: BaseEmbeddingReducer
    """The embedding reduction strategy to use."""

    similarity_measure: BaseEmbeddingSimilarity
    """The embedding similarity measure to use."""

    normalizer: BaseNormalizer
    """The SHAP value normalizer to use."""

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        mode: Mode = Mode.CONTEXTUAL,
        embedding_model: BaseExternalEmbedding | None = None,
        embedding_reducer: BaseEmbeddingReducer | None = None,
        similarity_measure: BaseEmbeddingSimilarity | None = None,
        normalizer: BaseNormalizer | None = None,
    ):
        """
        Initialize the SHAP base class.

        Args:
            mode: The SHAP mode, either STATIC or CONTEXTUAL. Used if no embedding_model is provided.
            embedding_model: The external embedding model to use. If provided, overrides mode.
            embedding_reducer: The embedding reduction strategy to use.
                Defaults to MeanReducer.
            similarity_measure: The embedding similarity measure to use.
                Defaults to CosineSimilarity.
            normalizer: The SHAP value normalizer to use.
                Defaults to PowerShiftNormalizer.
        """
        # validation
        __config = BaseShapConfig(
            mode=mode,
            embedding_model=embedding_model,
            embedding_reducer=embedding_reducer if embedding_reducer is not None else MeanReducer(),
            similarity_measure=similarity_measure if similarity_measure is not None else CosineSimilarity(),
            normalizer=normalizer if normalizer is not None else PowerShiftNormalizer(),
        )

        self.mode = __config.mode
        self.embedding_model = __config.embedding_model
        self.embedding_reducer = __config.embedding_reducer
        self.similarity_measure = __config.similarity_measure
        self.normalizer = __config.normalizer

    @abstractmethod
    def _generate_masks(self, n: int, device: torch.device, existing_masks: Tensor | None = None) -> Tensor:
        """
        Generate up to 2^n boolean masks of length n.

        Args:
            n: Length of the masks
            device: The device to create the masks on
            existing_masks: Optional existing masks to reuse.
        Returns:
            Tensor of shape [num_splits, n], dtype=torch.bool,
                where num_splits depends on the implementation
        """

    @abstractmethod
    def _calculate_shap_values(
        self,
        masks: Tensor,
        similarities: Tensor,
        device: torch.device,
    ) -> Tensor:
        """
        Calculate SHAP values based on similarity between base and masked embeddings.

        Args:
            masks (Tensor): 2D boolean tensor [num_masks, num_tokens],
                each row indicates which tokens are included in that mask.
            similarities (Tensor): 1D tensor [num_masks], similarity score for each mask.
            device: The device to create the SHAP values on.

        Returns:
            Tensor: 1D tensor [num_tokens] with SHAP values (NaN where base_mask=False).
        """

    def __get_masks(self, source_chat: BaseMllmChat) -> Tensor:
        """
        Generate masks of :attr:`input_tokens` using the True positions in :attr:`shap_values_mask`.

        Args:
            source_chat: The current chat state (without base response).
        Returns:
            Tensor of shape [num_splits, len(tokens)], dtype=torch.bool
                where num_splits depends on the implementation
        Raises:
            NoTokensToExplainError: If there are no tokens to explain in the provided chat or
                if mask has no True values.
            ValueError: If more masks are generated than possible.
        """
        mask = source_chat.shap_values_mask
        if not mask.any():
            raise NoTokensToExplainError("There are no tokens to explain in the provided chat.")

        target_length = source_chat.input_tokens_num
        logger.debug("Generating masks for target length %d using provided mask.", target_length)

        n = int(mask.sum().item())
        if n == 0:
            raise NoTokensToExplainError("Mask must have at least one True value.")
        max_masks = 2**n - 1
        logger.info(
            "Number of tokens for explainability: %d, (up to %d additional calls)",
            n,
            max_masks,
        )

        masks = self._generate_masks(n, device=source_chat.torch_device)
        if len(masks) > max_masks:
            raise ValueError("Generated more masks than possible.")

        # add base mask (all True)
        masks = torch.cat(
            [
                torch.ones((1, n), dtype=torch.bool, device=source_chat.torch_device),
                masks,
            ],
            dim=0,
        )

        # extend masks to full length
        return self.__prepare_final_masks(
            masks,
            target_length=target_length,
            mask=mask,
            device=source_chat.torch_device,
        )

    def __prepare_final_masks(self, splits: Tensor, target_length: int, mask: Tensor, device: torch.device) -> Tensor:
        """
        Prepare the final masks by setting masked positions according to splits
        and keeping unmasked positions always True.

        Args:
            splits: Tensor of shape [num_splits, num_masked], dtype=torch.bool
            target_length: Length of the final masks to be generated
            mask: 1D boolean tensor indicating which positions to split
            device: The device to create the masks on
        Returns:
            Tensor of shape [num_splits, len(tokens)], dtype=torch.bool
        """
        final_masks = torch.zeros((len(splits), target_length), dtype=torch.bool, device=device)

        # Set masked positions according to splits
        final_masks[:, mask] = splits
        # Keep unmasked positions always True
        final_masks[:, ~mask] = True

        # Filter out rows that have no True values (completely empty masks)
        # it is a case scenario when all tokens are taken into account for splitting
        valid_mask = final_masks.any(dim=1)
        final_masks = final_masks[valid_mask]

        # sort for deterministic output
        final_masks.sort(dim=0)
        return final_masks

    def __get_embeddings(self, chat: BaseMllmChat, model: BaseMllmModel) -> Tensor:
        """
        Get embeddings for the given chat state.

        Args:
            chat: The current chat state.
            model: The model instance.
        Returns:
            The embeddings tensor.
        """
        if self.embedding_model is not None:
            return self.embedding_model(chat=chat)

        if self.mode == Mode.STATIC:
            return model.get_static_embeddings(chat=chat)
        return model.get_contextual_embeddings(chat=chat)

    def __prepare_reduced_embeddings_tensor(
        self, base_size: int, response_chat: BaseMllmChat, model: BaseMllmModel
    ) -> Tensor:
        """
        Prepare the reduced embeddings tensor.

        Args:
            base_size: The base size for the reduced embeddings tensor.
            response_chat: The response chat instance (without earlier history).
            model: The model instance.
        Returns:
            The reduced embeddings tensor.
        """

        # prepare response embeddings
        reduced_response_embedding = self.embedding_reducer(
            self.__get_embeddings(
                chat=response_chat,
                model=model,
            )
        )
        reduced_embeddings: Tensor = torch.empty(
            (base_size, reduced_response_embedding.shape[0]),
            device=response_chat.torch_device,
            dtype=reduced_response_embedding.dtype,
        )
        reduced_embeddings[0] = reduced_response_embedding

        return reduced_embeddings

    def __read_cache(
        self, masks: Tensor, reduced_embeddings: Tensor, full_chat: BaseMllmChat
    ) -> tuple[Tensor, Tensor, int]:
        """
        Get or set the SHAP explainer cache in the full chat.

        Args:
            masks: The generated masks.
            reduced_embeddings: The reduced embeddings for each mask.
            full_chat: The full chat instance to get the cache from
                (with history and base response).
        Returns:
            A tuple containing:
            - The updated masks tensor.
            - The updated reduced embeddings tensor.
            - The starting index for new embeddings in reduced_embeddings.
        Raises:
            ValueError: If existing cache is invalid.
        """
        logger.debug("Getting or setting SHAP explainer cache for chat %s.", full_chat)

        start_idx: int = 1  # start after base response embedding
        cache: ExplainerCache | None = full_chat.shap
        if cache is not None:
            if cache.calculated_by != hash(self):
                raise ValueError("Existing SHAP cache was calculated by a different explainer instance.")
            if cache.chat != full_chat:
                raise ValueError("Existing SHAP cache is associated with a different chat instance.")
            if cache.reduced_embeddings is None:
                raise ValueError("Existing SHAP cache has no reduced embeddings stored.")
            if cache.masks is None:
                raise ValueError("Existing SHAP cache has no masks stored.")

            # Extend existing masks to match new masks size
            existing_masks = cache.extend_values(
                cache.masks,
                shape=(cache.masks.shape[0], masks.shape[1] - cache.masks.shape[1]),
                dim=1,
                fill_value=False,  # extend with False as those tokens were not considered for splitting
            )

            # remove any possible duplicates with existing masks
            removed_indices, extracted_indices = self.__deduplicate_masks(
                new_masks=masks,
                existing_masks=existing_masks,
            )
            del existing_masks

            # reorder masks so that removed are at the beginning
            removed_mask = torch.zeros(masks.shape[0], dtype=torch.bool, device=masks.device)
            removed_mask[removed_indices] = True
            masks = torch.cat(
                [masks[removed_indices], masks[~removed_mask]], dim=0
            )  # call first part with removed_indices to maintain order

            # save extracted embeddings from cache
            reduced_embeddings[start_idx : start_idx + extracted_indices.shape[0]] = (  # noqa: E203
                cache.reduced_embeddings[
                    extracted_indices
                ]  # maintain order, skip 1 as it is a 'base response' embedding
            )
            start_idx += extracted_indices.shape[0]

            del full_chat.shap
            del cache

            logger.info(
                "Deduplicated %d/%d masks using existing cache.",
                extracted_indices.shape[0],
                masks.shape[0] - 1,  # exclude base mask
            )

        return masks, reduced_embeddings, start_idx

    def __get_shap_values(
        self,
        masks: Tensor,
        reduced_embeddings: Tensor,
        source_chat: BaseMllmChat,
        full_chat: BaseMllmChat,
    ) -> tuple[Tensor, Tensor]:
        """
        Get SHAP values for the given mask.

        Args:
            mask: The mask to get SHAP values for.
            reduced_embeddings: The reduced embeddings for each mask.
            source_chat: The current chat state (without base response).
            full_chat: The full chat instance (with history and base response).
        Returns:
            A tuple containing:
            - The calculated SHAP values.
            - The normalized SHAP values.
        """
        shap_values_mask = source_chat.shap_values_mask

        # calculate similarities between original response embeddings
        similarities = self.similarity_measure(reduced_embeddings[0], reduced_embeddings)

        # Pre-allocate SHAP values with NaNs
        shap_values = torch.full_like(
            shap_values_mask,
            float("nan"),
            device=full_chat.torch_device,
            dtype=similarities.dtype,
        )

        # Calculate SHAP values only for relevant parts
        calculated_shap_values = self._calculate_shap_values(
            masks=masks[..., shap_values_mask],  # only pass relevant parts of masks
            similarities=similarities,
            device=full_chat.torch_device,
        )
        shap_values[shap_values_mask] = calculated_shap_values

        # Normalize only calculated SHAP values
        normalized_shap_values = shap_values.clone()
        normalized_shap_values[shap_values_mask] = self.normalizer(calculated_shap_values)

        return shap_values, normalized_shap_values

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __save_to_cache(
        self,
        full_chat: BaseMllmChat,
        masks: Tensor,
        reduced_embeddings: Tensor,
        shap_values: Tensor,
        normalized_shap_values: Tensor,
    ) -> None:
        """
        Save the SHAP explainer cache in the full chat.

        Args:
            full_chat: The full chat instance to save the cache for
                (with history and base response).
            masks: The generated masks.
            reduced_embeddings: The reduced embeddings for each mask.
            shap_values: The calculated SHAP values.
            normalized_shap_values: The normalized SHAP values.
        Raises:
            ValueError: If cache already exists for the provided chat.
        """
        logger.debug("Saving SHAP explainer cache for chat %s.", full_chat)
        if full_chat.shap is not None:
            raise ValueError("SHAP cache already exists for the provided chat.")

        cache = ExplainerCache(
            calculated_by=hash(self),
            chat=full_chat,
        )

        cache.n = masks.shape[1]

        cache.values = shap_values
        cache.normalized_values = normalized_shap_values

        cache.reduced_embeddings = reduced_embeddings
        cache.masks = masks

        full_chat.shap = cache

    def __deduplicate_masks(
        self,
        new_masks: Tensor,
        existing_masks: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Remove masks from new_masks that already exist in existing_masks.

        Args:
            new_masks: Tensor of shape [num_new_masks, num_tokens], dtype=torch.bool
            existing_masks: Tensor of shape [num_existing_masks, num_tokens], dtype=torch.bool
            return_removed_indices: Whether to return the indices of removed masks
        Returns:
            A tuple containing:
            - Tensor of shape [num_removed_masks], dtype=torch.long
                with indices of the removed masks in new_masks.
            - Tensor of shape [num_removed_masks], dtype=torch.bool
                with indices of used masks in existing_masks.
        """
        existing_set: dict[tuple[Any, ...], int] = {tuple(row.tolist()): i for i, row in enumerate(existing_masks)}
        removed_indices: list[int] = []
        extracted_indices: list[int] = []

        for i, mask in enumerate(new_masks):
            key = tuple(mask.tolist())
            if key in existing_set:
                removed_indices.append(i)
                extracted_indices.append(existing_set[key])

        removed_indices_tensor = torch.tensor(removed_indices, device=new_masks.device, dtype=torch.long)
        extracted_indices_tensor = torch.tensor(extracted_indices, device=new_masks.device, dtype=torch.long)

        return removed_indices_tensor, extracted_indices_tensor

    # keep the logic in one method for readability
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    def __call__(
        self,
        model: BaseMllmModel,
        source_chat: BaseMllmChat,
        response_chat: BaseMllmChat,
        full_chat: BaseMllmChat,
        progress_bar: bool = True,
        verbose: bool = False,
        **generate_kwargs: Any,
    ) -> list[tuple[Tensor, BaseMllmChat, BaseMllmChat, BaseMllmChat, Tensor] | None] | None:
        """
        Generate splits of the input tokens in the chat state.

        Args:
            model: The model instance.
            source_chat: Chat to get explained (without base response).
            response_chat: The chat with base response to compare against (without earlier history).
            full_chat: The full chat including both model response and earlier history.
                It will be set with chat values and searched for existing SHAP cache.
            progress_bar: Whether to display a progress bar during processing.
            verbose: Whether to save data generated during processing.
            generate_kwargs: Additional keyword arguments for the model's generate method.
        Returns:
            If verbose is True, returns the history of chats and masks used during explanation.
                History has entries of the form
                (mask, masked_source_chat, masked_response_chat, masked_full_chat, embeddings) or
                None if corresponding chat was skipped due to all text tokens being filtered out
                or its extraction from cache.
            If verbose is False, returns None.
        Raises:
            NoTokensToExplainError: If there are no tokens to explain in the provided chat.
            NotEnoughTokensToExplainError: If there are not enough tokens to explain after filtering out
                empty chats.
            ValueError: If existing cache is invalid.
        """
        __config = BaseShapCallConfig(
            model=model,
            source_chat=source_chat,
            response_chat=response_chat,
            full_chat=full_chat,
            progress_bar=progress_bar,
            verbose=verbose,
        )

        # remove shap caches as they won't be needed during processing
        del __config.source_chat.shap
        del __config.response_chat.shap

        # prepare masks dependent on approximation method
        masks = self.__get_masks(source_chat=__config.source_chat)

        # prepare embeddings tensor
        reduced_embeddings = self.__prepare_reduced_embeddings_tensor(
            base_size=len(masks),
            response_chat=__config.response_chat,  # we want to compare to response chat only
            model=__config.model,
        )

        # read cache if available
        masks, reduced_embeddings, start_idx = self.__read_cache(
            masks=masks,
            reduced_embeddings=reduced_embeddings,
            full_chat=__config.full_chat,
        )

        # pre-allocate all variables
        masked_response_chat: BaseMllmChat
        masked_full_chat: BaseMllmChat | None = None
        chats_skipped_indices: list[int] = []
        history: list[tuple[Tensor, BaseMllmChat, BaseMllmChat, BaseMllmChat, Tensor] | None] | None = (
            None if not verbose else [None] * len(masks)
        )

        # start_idx = 64
        gen = tqdm(masks[start_idx:], desc="Calculating SHAP values") if progress_bar else masks[start_idx:]
        for i, mask in enumerate(gen, start=start_idx):
            logger.debug("Processing mask %s", mask)

            # prepare chat containing current scope history
            try:
                masked_chat = type(source_chat).from_chat(
                    mask=mask,
                    chat=source_chat,
                )
            except AllTextTokensFilteredOutError:
                logger.warning("All text tokens filtered out for mask %d", i)
                chats_skipped_indices.append(i)
                continue

            # generate response for masked chat
            t0 = time()
            r = model.generate(chat=masked_chat, keep_history=verbose, **generate_kwargs)
            logger.debug("Generation took %.2f seconds", time() - t0)
            if verbose:
                masked_response_chat, masked_full_chat = r  # type: ignore[misc]
            else:
                masked_response_chat = r  # type: ignore[assignment]

            # get embeddings for the response
            embeddings = self.__get_embeddings(
                chat=masked_response_chat,
                model=model,
            )
            # apply embedding reduction
            reduced_embedding = self.embedding_reducer(embeddings)
            reduced_embeddings[i] = reduced_embedding

            if verbose:
                # here history is not None and is to be populated
                history[i] = (  # type: ignore[index]
                    mask,
                    masked_chat,
                    masked_response_chat,
                    cast(BaseMllmChat, masked_full_chat),
                    reduced_embedding,
                )
            else:
                # cleanup to avoid memory leaks
                del masked_chat
                del masked_response_chat
                gc.collect()

        # edge case - all chats were empty after filtering yet shap_values_mask had True values
        # this can happen only if shap_values_mask has one True value
        # for simplicity we just raise an error here. - 1 because
        # masks will always have at least the base mask that cannot ever
        # be in chats_skipped_indices
        if masks.shape[0] - 1 <= len(chats_skipped_indices):
            raise NotEnoughTokensToExplainError(
                "Not enough tokens to explain after filtering out empty chats. "
                "Ensure that shap_values_mask has at least two True values.",
            )

        # filter out skipped due to all text tokens filtered out
        if chats_skipped_indices:
            filtering_mask = torch.ones(masks.shape[0], dtype=torch.bool, device=masks.device)
            filtering_mask[chats_skipped_indices] = False

            masks = masks[filtering_mask]
            reduced_embeddings = reduced_embeddings[filtering_mask]

            logger.info(
                "Skipped %d chats due to all text tokens being filtered out.",
                len(chats_skipped_indices),
            )

        # calculate SHAP values
        shap_values, normalized_shap_values = self.__get_shap_values(
            masks=masks,
            reduced_embeddings=reduced_embeddings,
            source_chat=__config.source_chat,
            full_chat=__config.full_chat,
        )

        # cache results
        self.__save_to_cache(
            full_chat=__config.full_chat,
            masks=masks,
            reduced_embeddings=reduced_embeddings,
            shap_values=shap_values,
            normalized_shap_values=normalized_shap_values,
        )

        return history

    def __hash__(self) -> int:
        """
        Get the hash of the explainer instance.

        Returns:
            The hash value.
        """
        return hash(
            (
                self.mode,
                self.embedding_reducer,
                self.similarity_measure,
                self.normalizer,
            )
        )

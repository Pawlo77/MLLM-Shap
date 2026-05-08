"""Base class for SHAP-based explanations."""

from abc import ABC, abstractmethod
from logging import Logger
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from ...utils.other import extend_tensor
from ...connectors.base.explainer_cache import ExplainerCache
from ...connectors.base.model import BaseMllmModel
from ...connectors.base.model_response import ModelResponse
from ...connectors.base.chat import BaseMllmChat
from ...utils.logger import get_logger
from ..embeddings import MeanReducer
from ..enums import Mode
from ..normalizers import PowerShiftNormalizer
from ..similarity import CosineSimilarity
from ._validators import BaseShapCallConfig, BaseShapConfig
from .embeddings import BaseEmbeddingReducer, BaseExternalEmbedding
from .normalizers import BaseNormalizer
from .similarity import BaseEmbeddingSimilarity
from ._masks_manager import MasksManager
from ._mask_generator import MaskGenerator
from ._cache_manager import CacheManager
from ._generate_responses import generate_responses
from ..pipeline import ExplainContext, ExplainState, PipelinePreset
from ..pipeline.stages import InsufficientMasksError
from ..pipeline.stages.sampling_adapter import (
    build_masks_generator,
    run_sampling_generation,
)

if TYPE_CHECKING:
    from ..core.telemetry import TelemetryProbe
    from ...observability.sink import ObservabilitySink

logger: Logger = get_logger(__name__)


class NotEnoughTokensToExplainError(Exception):
    """Raised when there are not enough tokens to explain in the chat."""


class BaseShapExplainer(ABC):
    """Base class for SHAP-based explanations."""

    _tqdm_desc: str = "SHAP"
    """Description displayed in the tqdm progress bar."""

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

    allow_mask_duplicates: bool
    """Whether to allow duplicate masks during generation."""

    total_n_calls: int = 0
    """Total number of MLLM calls made for last explanation."""

    last_observability_sink: "ObservabilitySink | None" = None
    """Sink used for the most recent call (if observability was enabled)."""

    _first_call: bool
    """Indicates if it's the first call to generate masks."""

    def __init__(
        self,
        mode: Mode = Mode.CONTEXTUAL,
        embedding_model: BaseExternalEmbedding | None = None,
        embedding_reducer: BaseEmbeddingReducer | None = None,
        similarity_measure: BaseEmbeddingSimilarity | None = None,
        normalizer: BaseNormalizer | None = None,
        allow_mask_duplicates: bool = False,
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
            allow_mask_duplicates: Whether to allow duplicate masks during generation.
        """
        # validation
        __config = BaseShapConfig(
            mode=mode,
            embedding_model=embedding_model,
            embedding_reducer=embedding_reducer
            if embedding_reducer is not None
            else MeanReducer(),
            similarity_measure=similarity_measure
            if similarity_measure is not None
            else CosineSimilarity(),
            normalizer=normalizer if normalizer is not None else PowerShiftNormalizer(),
            allow_mask_duplicates=allow_mask_duplicates,
        )

        self.mode = __config.mode
        self.embedding_model = __config.embedding_model
        self.embedding_reducer = __config.embedding_reducer
        self.similarity_measure = __config.similarity_measure
        self.normalizer = __config.normalizer
        self.allow_mask_duplicates = __config.allow_mask_duplicates

    @abstractmethod
    def _get_next_split(
        self,
        n: int,
        device: torch.device,
        generated_masks_num: int,
        existing_masks: list[Tensor] | None = None,
    ) -> Tensor | None:
        """
        Get next split to evaluate.

        Args:
            n: Length of the splits
            device: The device to create the masks on
            generated_masks_num: Number of masks generated so far
            existing_masks: List of existing masks
        Returns:
            Tensor of shape [1, n], dtype=torch.bool, representing the next split to evaluate
                or None if no more splits are to be generated.
        """

    @abstractmethod
    def _get_num_splits(self, n: int) -> int:
        """
        Determine the number of masks to generate based on num_samples and fraction.

        Args:
            n: Length of the splits
        Returns:
            Number of masks to generate.
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
                The first mask (index 0) represents the base mask with all tokens included
                (all True values).
            similarities (Tensor): 1D tensor [num_masks], similarity score for each mask.
            device: The device to create the SHAP values on.
        Returns:
            Tensor: 1D tensor [num_tokens] with SHAP values (NaN where base_mask=False).
        """

    def _initialize_state(self) -> None:
        """
        Initialize internal state before starting mask generation.
        """
        self.total_n_calls = 0
        self._first_call = True
        self.last_observability_sink = None

    def _get_masks_generator(
        self,
        mask_manager: MasksManager,
        device: torch.device,
        masks: list[Tensor],
        allow_full_or_empty: bool = False,
    ) -> MaskGenerator:
        """
        Generator that yields masks one by one.

        Args:
            mask_manager: The masks manager instance.
            device: The device to create the masks on.
            masks: List of existing masks.
        Returns:
            A generator yielding tuples of (mask, mask_hash).
        """
        return build_masks_generator(
            get_next_split=self._get_next_split,
            get_num_splits=self._get_num_splits,
            mask_manager=mask_manager,
            device=device,
            masks=masks,
            allow_mask_duplicates=self.allow_mask_duplicates,
            allow_full_or_empty=allow_full_or_empty,
        )

    def _generate_step(
        self,
        mask_manager: MasksManager,
        device: torch.device,
        masks: list[Tensor],
        tqdm_bar: Any | None = None,
        tqdm_desc: str = "Calculating SHAP values",
        **generate_kwargs: Any,
    ) -> tuple[
        int, list[tuple[Tensor, int, BaseMllmChat | None, ModelResponse]] | None
    ]:
        """
        Generate a step of masks and get model responses.

        Args:
            mask_manager: The masks manager instance.
            device: The device to create the masks on.
            masks: List of existing masks.
            tqdm_bar: Optional external tqdm bar to update instead of creating a new one.
            tqdm_desc: Description to display in the tqdm progress bar.
            generate_kwargs: Additional keyword arguments for the model's generate method.
        Returns:
            A tuple containing:
            - Number of chats skipped due to being empty.
            - History of chats and masks used during explanation.
        """
        result, generated_masks = run_sampling_generation(
            get_next_split=self._get_next_split,
            get_num_splits=self._get_num_splits,
            mask_manager=mask_manager,
            device=device,
            masks=masks,
            allow_mask_duplicates=self.allow_mask_duplicates,
            allow_full_or_empty=False,
            logger=logger,
            tqdm_bar=tqdm_bar,
            tqdm_desc=tqdm_desc,
            generate_responses_fn=generate_responses,
            get_masks_generator=self._get_masks_generator,
            **generate_kwargs,
        )
        self.total_n_calls = generated_masks
        return result

    def _get_similarities(
        self, responses: list[ModelResponse], model: BaseMllmModel
    ) -> Tensor:
        """
        Get similarities between the base response and other responses.

        Args:
            responses: The model responses to compare.
            model: The model instance.
        Returns:
            A tensor containing the similarities.
        """
        if self.similarity_measure.operates_on_embeddings:
            embeddings = self.__get_embeddings(
                responses=responses,
                model=model,
            )
            return self.similarity_measure(base=embeddings[0], other=embeddings)

        return self.similarity_measure(base=responses[0], other=responses)

    def _get_shap_values(
        self,
        model: BaseMllmModel,
        masks: Tensor,
        responses: list[ModelResponse],
        source_chat: BaseMllmChat,
        device: torch.device,
        similarities: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Get SHAP values for the given mask.

        Args:
            model: The model instance.
            masks: 2D boolean tensor [num_masks, num_tokens],
                each row indicates which tokens are included in that mask.
            responses: The model responses corresponding to the masks.
            source_chat: The source chat instance used to search for
                external group ids.
            device: The device to create the SHAP values on.
            similarities: Precomputed similarities between base and masked responses.
                If None, will be computed.
        Returns:
            A tuple containing:
            - The calculated SHAP values.
            - The normalized SHAP values.
        """
        shap_values_mask = source_chat.shap_values_mask
        if similarities is None:
            similarities = self._get_similarities(responses=responses, model=model)

        # Pre-allocate SHAP values with NaNs
        shap_values = torch.full_like(
            shap_values_mask,
            float("nan"),
            device=device,
            dtype=similarities.dtype,
        )

        # Calculate SHAP values only for relevant parts
        calculated_shap_values = self._calculate_shap_values(
            masks=masks[..., shap_values_mask],  # only pass relevant parts of masks
            similarities=similarities,
            device=device,
        )
        shap_values[shap_values_mask] = calculated_shap_values

        # Normalize only calculated SHAP values
        normalized_shap_values = shap_values.clone()
        normalized_shap_values[shap_values_mask] = self.normalizer(
            calculated_shap_values
        )

        # duplicate if external group ids are used
        if source_chat.external_group_ids is not None:
            for group_id, group_shap_value, group_normalized_shap_value in zip(
                source_chat.external_group_ids[
                    source_chat.external_group_ids_first_positions
                ],
                shap_values[source_chat.external_group_ids_first_positions],
                normalized_shap_values[source_chat.external_group_ids_first_positions],
            ):
                mask = source_chat.external_group_ids == group_id
                shap_values[mask] = group_shap_value
                normalized_shap_values[mask] = group_normalized_shap_value

        return shap_values, normalized_shap_values

    def _save_to_cache(
        self,
        chat: BaseMllmChat,
        source_chat: BaseMllmChat,
        responses: list[ModelResponse],
        masks: Tensor,
        shap_values: Tensor,
        normalized_shap_values: Tensor,
    ) -> None:
        """
        Save the SHAP explainer cache in the full chat.

        Args:
            chat: The chat instance to save the cache for.
            source_chat: The original chat instance from which SHAP values were derived.
            responses: The model responses used for SHAP calculations.
            masks: The masks used for SHAP calculations.
            shap_values: The SHAP values calculated.
            normalized_shap_values: The normalized SHAP values calculated.
        Raises:
            ValueError: If cache already exists for the provided chat.
        """
        logger.debug("Saving SHAP explainer cache for chat %s.", chat)
        if chat.cache is not None:
            raise ValueError("SHAP cache already exists for the provided chat.")

        # translate it for reference to group ids
        shap_values_mask = source_chat.translate_groups_ids_mask(
            source_chat.shap_values_mask
        )
        # extend mask with False to match new response length
        shap_values_mask = extend_tensor(
            shap_values_mask,
            target_length=chat.input_tokens_num,
            fill_value=False,
        )

        chat.cache = ExplainerCache.create(
            chat=chat,
            explainer_hash=hash(self),
            responses=responses,
            masks=masks,
            values=shap_values,
            normalized_values=normalized_shap_values,
            shap_values_mask=shap_values_mask,
        )

    def __get_embeddings(
        self, responses: list[ModelResponse], model: BaseMllmModel
    ) -> Tensor:
        """
        Get embeddings for the given chat state.

        Args:
            responses: The model responses to get embeddings for.
            chat: The current chat state.
        Returns:
            The embeddings tensor.
        """
        if self.embedding_model is not None:
            return self.embedding_reducer(self.embedding_model(responses=responses))

        if self.mode == Mode.STATIC:
            return self.embedding_reducer(
                model.get_static_embeddings(responses=responses)
            )
        return self.embedding_reducer(
            model.get_contextual_embeddings(responses=responses)
        )

    # keep the logic in one method for readability
    def __call__(
        self,
        model: BaseMllmModel,
        source_chat: BaseMllmChat,
        response: ModelResponse,
        progress_bar: bool = True,
        verbose: bool = False,
        n_generator_jobs: int = 1,
        probe: "TelemetryProbe | None" = None,
        observability_enabled: bool = False,
        observability_sink: "ObservabilitySink | None" = None,
        observability_run_id: str | None = None,
        **generate_kwargs: Any,
    ) -> list[tuple[Tensor, int, BaseMllmChat | None, ModelResponse]] | None:
        """
        Generate splits of the input tokens in the chat state.

        Args:
            model: The model instance.
            source_chat: Chat to get explained (without base response).
            response: The model response generated from source_chat.
            progress_bar: Whether to display a progress bar during processing.
            verbose: Whether to save data generated during processing.
            n_generator_jobs: Number of parallel calls to the model's generate method.
            probe: Optional telemetry probe for instrumenting mask generation and caching.
                If None, no telemetry is collected (zero overhead). Defaults to None.
            observability_enabled: If True and no sink is provided, auto-create
                an in-memory observability sink for stage events/spans.
            observability_sink: Optional structured observability sink. If provided,
                pipeline events/spans are emitted to this sink.
            observability_run_id: Optional run id used for emitted observability records.
            generate_kwargs: Additional keyword arguments for the model's generate method.
        Returns:
            If verbose is True, returns the history of chats and masks used during explanation.
                History has entries of the form (mask, mask_hash, masked_chat, model_response).
                If cache was used, masked_chat will be None.
            If verbose is False, returns None.
        Raises:
            NotEnoughTokensToExplainError: If there are not enough tokens to explain after filtering out
                empty chats.
            ValueError: If existing cache is invalid.
        """
        __config = BaseShapCallConfig(
            model=model,
            source_chat=source_chat,
            response=response,
            progress_bar=progress_bar,
            verbose=verbose,
        )
        self._initialize_state()

        # validated within BaseShapCallConfig
        response_chat: BaseMllmChat = __config.response.chat
        source_chat = __config.source_chat
        device = source_chat.torch_device

        active_observability_sink = observability_sink
        if active_observability_sink is None and observability_enabled:
            from ...observability.sink import InMemoryObservabilitySink

            active_observability_sink = InMemoryObservabilitySink()
        self.last_observability_sink = active_observability_sink

        pipeline_params = dict(generate_kwargs)
        if active_observability_sink is not None:
            pipeline_params["observability_sink"] = active_observability_sink
        if observability_run_id is not None:
            pipeline_params["run_id"] = observability_run_id

        pipeline_context = ExplainContext(
            model=__config.model,
            source_chat=source_chat,
            response_chat=response_chat,
            base_response=__config.response,
            device=device,
            params=MappingProxyType(pipeline_params),
        )
        pipeline_state = ExplainState()
        pipeline_state.add_metadata("explainer_hash", hash(self))

        pipeline = PipelinePreset(
            get_next_split=self._get_next_split,
            get_num_splits=self._get_num_splits,
            get_similarities=self._get_similarities,
            get_shap_values=self._get_shap_values,
            save_to_cache=self._save_to_cache,
            allow_mask_duplicates=self.allow_mask_duplicates,
            allow_full_or_empty=False,
            n_generator_jobs=n_generator_jobs,
            progress_bar=__config.progress_bar,
            verbose=__config.verbose,
            tqdm_desc=self._tqdm_desc,
            generate_kwargs=dict(generate_kwargs),
            masks_manager_factory=MasksManager,
            cache_manager_factory=CacheManager,
            generate_step=self._generate_step,
            include_sampling=True,
            include_attribution=True,
            include_finalize=True,
        ).build()
        try:
            pipeline.run(context=pipeline_context, state=pipeline_state, probe=probe)
        except InsufficientMasksError as e:
            raise NotEnoughTokensToExplainError(str(e)) from e

        masks = pipeline_state.masks
        history = pipeline_state.history

        cache_extracted = int(pipeline_state.metadata.get("cache_extracted", 0))
        if cache_extracted > 0:
            logger.info(
                "Deduplicated %d/%d masks using existing cache.",
                cache_extracted,
                len(masks) - 1,  # exclude base mask
            )

        self.total_n_calls = int(
            pipeline_state.metadata.get("generated_masks", self.total_n_calls)
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

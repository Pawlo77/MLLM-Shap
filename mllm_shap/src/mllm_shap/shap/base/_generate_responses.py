"""Generate model responses for masked SHAP chats."""

from __future__ import annotations

from dataclasses import dataclass
from logging import Logger
from threading import Lock, Thread
from time import time
from typing import Any

from torch import Tensor
from tqdm import tqdm as standard_tqdm
from tqdm.auto import tqdm

from ...connectors.base.chat import AllTextTokensFilteredOutError, BaseMllmChat
from ...connectors.base.model import BaseMllmModel
from ...connectors.base.model_response import ModelResponse
from ...errors import WorkerExecutionError
from ._mask_generator import MaskGenerator
from ...utils.logger import get_logger
from ._cache_manager import CacheManager

logger: Logger = get_logger(__name__)


@dataclass(frozen=True)
class _GenerationContext:
    """Context required to process masks.

    Attributes:
        source_chat: Chat used as the source for masked clones.
        model: Connector model used to generate responses.
        cache_manager: Cache manager used for mask-hash lookups.
        verbose: Flag that controls whether to keep chat history.
        generate_kwargs: Keyword arguments forwarded to model generation.
    """

    source_chat: BaseMllmChat
    model: BaseMllmModel
    cache_manager: CacheManager
    verbose: bool
    generate_kwargs: dict[str, Any]


@dataclass
class _GenerationAccumulator:
    """Accumulator for generated artifacts.

    Attributes:
        masks: Output list collecting masks that were evaluated.
        responses: Output list collecting model responses.
        history: Optional verbose history entries.
    """

    masks: list[Tensor]
    responses: list[ModelResponse]
    history: list[tuple[Tensor, int, BaseMllmChat | None, ModelResponse]] | None

    def append(
        self,
        *,
        mask: Tensor,
        mask_hash: int,
        masked_chat: BaseMllmChat | None,
        model_response: ModelResponse,
    ) -> None:
        """Store one generated result.

        Args:
            mask: Evaluated mask.
            mask_hash: Hash for the evaluated mask.
            masked_chat: Masked chat used for generation, or ``None`` if cached.
            model_response: Generated model response.
        """
        self.masks.append(mask)
        self.responses.append(model_response)
        if self.history is not None:
            self.history.append((mask, mask_hash, masked_chat, model_response))


class _MaskProcessor:
    """Cache-aware processor for one SHAP mask."""

    def __init__(self, context: _GenerationContext) -> None:
        self._context = context

    def process(
        self, *, mask: Tensor, mask_hash: int, i: int
    ) -> tuple[BaseMllmChat | None, ModelResponse]:
        """Process one mask.

        Args:
            mask: Mask tensor to evaluate.
            mask_hash: Precomputed mask hash.
            i: Iteration index used in logs.

        Returns:
            Tuple with masked chat and model response.
        """
        return _process_mask(
            mask=mask,
            mask_hash=mask_hash,
            source_chat=self._context.source_chat,
            model=self._context.model,
            cache_manager=self._context.cache_manager,
            verbose=self._context.verbose,
            i=i,
            **self._context.generate_kwargs,
        )


def generate_responses(
    masks: list[Tensor],
    responses: list[ModelResponse],
    gen: MaskGenerator,
    source_chat: BaseMllmChat,
    model: BaseMllmModel,
    cache_manager: CacheManager,
    n_generator_jobs: int = 1,
    progress_bar: bool = True,
    verbose: bool = False,
    **generate_kwargs: dict[str, Any],
) -> tuple[int, list[tuple[Tensor, int, BaseMllmChat | None, ModelResponse]] | None]:
    """
    Generate model responses for all masks.

    Args:
        masks: List to store generated masks.
        responses: List to store generated model responses.
        gen: Generator yielding tuples of (mask, mask_hash).
        source_chat: The original chat object.
        model: The model to generate responses from.
        cache_manager: The cache manager to store/retrieve responses.
        n_generator_jobs: Number of parallel jobs to use for generation.
        progress_bar: Whether to display a progress bar.
        verbose: Whether to keep full history in the chat.
        generate_kwargs: Additional arguments for the model's generate method.
    Returns:
        Number of chats skipped due to all text tokens being filtered out
            and optionally the history of generated responses.
    """
    if n_generator_jobs > 1:
        return _generate_responses_multi(
            masks=masks,
            responses=responses,
            gen=gen,
            source_chat=source_chat,
            model=model,
            cache_manager=cache_manager,
            n_generator_jobs=n_generator_jobs,
            progress_bar=progress_bar,
            verbose=verbose,
            **generate_kwargs,
        )
    return _generate_responses_single(
        masks=masks,
        responses=responses,
        gen=gen,
        source_chat=source_chat,
        model=model,
        cache_manager=cache_manager,
        progress_bar=progress_bar,
        verbose=verbose,
        **generate_kwargs,
    )


def _process_mask(
    mask: Tensor,
    mask_hash: int,
    source_chat: BaseMllmChat,
    model: BaseMllmModel,
    cache_manager: CacheManager,
    verbose: bool,
    i: int,
    **generate_kwargs: dict[str, Any],
) -> tuple[BaseMllmChat | None, ModelResponse]:
    """
    Process a single mask: check cache or generate new response.

    Args:
        mask: The mask tensor to process.
        mask_hash: The hash of the mask for caching.
        source_chat: The original chat object.
        model: The model to generate responses from.
        cache_manager: The cache manager to store/retrieve responses.
        verbose: Whether to keep full history in the chat.
        i: The index of the current mask being processed.
        generate_kwargs: Additional arguments for the model's generate method.
    Raises:
        AllTextTokensFilteredOutError: If all text tokens are filtered out for the given mask
    Returns:
        A tuple of the masked chat (or None if from cache) and the model response.
    """
    logger.debug("Processing mask %s", mask)

    # read result from cache
    if cache_manager.contains(mask_hash=mask_hash):
        logger.debug("%d: Entry extracted from cache", i)

        masked_chat = None
        model_response = cache_manager.extract(mask_hash=mask_hash)
    # generate new response
    else:
        # prepare chat containing current scope history
        try:
            masked_chat = type(source_chat).from_chat(
                mask=mask,
                chat=source_chat,
            )
        except AllTextTokensFilteredOutError as e:
            logger.warning(
                "All text tokens were filtered out for mask %d, skipping.",
                i,
            )
            raise e

        # generate response for masked chat
        t0 = time()
        model_response = model.generate(
            chat=masked_chat,
            keep_history=verbose,
            **generate_kwargs,
        )
        logger.debug("%d: Generation took %.2f seconds", i, time() - t0)

    return masked_chat, model_response


def _create_accumulator(
    *,
    masks: list[Tensor],
    responses: list[ModelResponse],
    verbose: bool,
) -> _GenerationAccumulator:
    """Create output accumulator.

    Args:
        masks: Target list for masks.
        responses: Target list for responses.
        verbose: Whether history should be collected.

    Returns:
        Prepared accumulator instance.
    """
    return _GenerationAccumulator(
        masks=masks,
        responses=responses,
        history=[] if verbose else None,
    )


def _generate_responses_multi(
    masks: list[Tensor],
    responses: list[ModelResponse],
    gen: MaskGenerator,
    source_chat: BaseMllmChat,
    model: BaseMllmModel,
    cache_manager: CacheManager,
    n_generator_jobs: int = 1,
    progress_bar: bool = True,
    verbose: bool = False,
    **generate_kwargs: dict[str, Any],
) -> tuple[int, list[tuple[Tensor, int, BaseMllmChat | None, ModelResponse]] | None]:
    """Generate model responses for all masks in parallel using multiple threads."""
    context = _GenerationContext(
        source_chat=source_chat,
        model=model,
        cache_manager=cache_manager,
        verbose=verbose,
        generate_kwargs=generate_kwargs,
    )
    processor = _MaskProcessor(context=context)
    accumulator = _create_accumulator(
        masks=masks,
        responses=responses,
        verbose=verbose,
    )
    iterable_gen = enumerate(
        standard_tqdm(gen, desc="Calculating SHAP values") if progress_bar else gen
    )

    chats_skipped = 0
    error_flag = False
    lock = Lock()

    def worker() -> None:
        """Worker function for generating masks in parallel."""
        nonlocal chats_skipped, error_flag

        while True:
            try:
                with lock:
                    if error_flag:
                        return

                    try:
                        i, (mask, mask_hash) = next(iterable_gen)
                    except StopIteration:
                        return

                try:
                    masked_chat, model_response = processor.process(
                        mask=mask, mask_hash=mask_hash, i=i
                    )
                except AllTextTokensFilteredOutError:
                    with lock:
                        chats_skipped += 1
                    continue

                with lock:
                    accumulator.append(
                        mask=mask,
                        mask_hash=mask_hash,
                        masked_chat=masked_chat,
                        model_response=model_response,
                    )

                if not verbose:
                    # cleanup large refs; avoid forcing global GC in hot loop
                    del masked_chat
                    del model_response

            except Exception as e:
                logger.error("Error in SHAP explainer worker: %s", e, exc_info=True)
                with lock:
                    error_flag = True
                return

    # run workers
    workers = [Thread(target=worker) for _ in range(n_generator_jobs)]
    for worker_thread in workers:
        worker_thread.start()
    for worker_thread in workers:
        worker_thread.join()
    if error_flag:
        raise WorkerExecutionError("Error occurred in SHAP explainer worker thread.")

    return chats_skipped, accumulator.history


def _generate_responses_single(
    masks: list[Tensor],
    responses: list[ModelResponse],
    gen: MaskGenerator,
    source_chat: BaseMllmChat,
    model: BaseMllmModel,
    cache_manager: CacheManager,
    progress_bar: bool = True,
    verbose: bool = False,
    **generate_kwargs: dict[str, Any],
) -> tuple[int, list[tuple[Tensor, int, BaseMllmChat | None, ModelResponse]] | None]:
    """Generate model responses for all masks sequentially."""
    context = _GenerationContext(
        source_chat=source_chat,
        model=model,
        cache_manager=cache_manager,
        verbose=verbose,
        generate_kwargs=generate_kwargs,
    )
    processor = _MaskProcessor(context=context)
    accumulator = _create_accumulator(
        masks=masks,
        responses=responses,
        verbose=verbose,
    )
    iterable_gen = enumerate(
        tqdm(gen, desc="Calculating SHAP values") if progress_bar else gen
    )

    chats_skipped = 0

    for i, (mask, mask_hash) in iterable_gen:
        try:
            masked_chat, model_response = processor.process(
                mask=mask, mask_hash=mask_hash, i=i
            )
        except AllTextTokensFilteredOutError:
            chats_skipped += 1
            continue

        accumulator.append(
            mask=mask,
            mask_hash=mask_hash,
            masked_chat=masked_chat,
            model_response=model_response,
        )

        if not verbose:
            # cleanup large refs; avoid forcing global GC in hot loop
            del masked_chat
            del model_response

    return chats_skipped, accumulator.history

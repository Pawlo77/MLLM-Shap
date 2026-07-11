"""Generate model responses for masked SHAP chats."""

from dataclasses import dataclass
from logging import Logger
from queue import Empty, Queue
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
    """Chat used as the source for masked clones. This chat should contain the full conversation history and will be masked according to the generated masks to create the inputs for the model."""
    model: BaseMllmModel
    """Connector model used to generate responses."""
    cache_manager: CacheManager
    """Cache manager used for mask-hash lookups."""
    verbose: bool
    """Flag that controls whether to keep chat history."""
    generate_kwargs: dict[str, Any]
    """Keyword arguments forwarded to model generation."""


@dataclass
class _GenerationAccumulator:
    """Accumulator for generated artifacts.

    Attributes:
        masks: Output list collecting masks that were evaluated.
        responses: Output list collecting model responses.
        history: Optional verbose history entries.
    """

    masks: list[Tensor]
    """Output list collecting masks that were evaluated. This list is mutated in-place and returned as part of the final pipeline state."""
    responses: list[ModelResponse]
    """Output list collecting model responses. This list is mutated in-place and returned as part of the final pipeline state."""
    history: list[tuple[Tensor, int, BaseMllmChat | None, ModelResponse]] | None
    """Optional verbose history entries. If the verbose flag is set, this list will collect tuples of (mask, mask_hash, masked_chat, model_response)
    for each processed mask. This can be used for debugging and analysis purposes, but may consume a large amount of memory
    if the number of masks is large or if the model responses are large. If verbose is False, this will be set to None and no history will be collected."""

    def append(
        self,
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
        """Initialize the processor with immutable generation context."""
        self._context = context

    def process(
        self, mask: Tensor, mask_hash: int, i: int
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
    tqdm_bar: tqdm | None = None,
    tqdm_desc: str = "Calculating SHAP values",
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
        tqdm_bar: Optional external tqdm bar to update instead of creating a new one.
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
            tqdm_bar=tqdm_bar,
            tqdm_desc=tqdm_desc,
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
        tqdm_bar=tqdm_bar,
        tqdm_desc=tqdm_desc,
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
        probe = cache_manager._probe
        if probe is None:
            model_response = model.generate(
                chat=masked_chat,
                keep_history=verbose,
                **generate_kwargs,
            )
        else:
            with probe.timing("model"):
                model_response = model.generate(
                    chat=masked_chat,
                    keep_history=verbose,
                    **generate_kwargs,
                )
        logger.debug("%d: Generation took %.2f seconds", i, time() - t0)

    return masked_chat, model_response


def _report_generation_metrics(
    cache_hits: int,
    cache_misses: int,
    chats_skipped: int,
    model_elapsed_ms: float,
    cache_manager: CacheManager,
) -> None:
    """Log generation summary and emit probe metrics when available."""
    total_processed = cache_hits + cache_misses
    logger.info(
        "Generation stats: processed=%d cache_hits=%d cache_misses=%d skipped_filtered=%d model_elapsed_ms=%.2f",
        total_processed,
        cache_hits,
        cache_misses,
        chats_skipped,
        model_elapsed_ms,
    )

    probe = cache_manager._probe
    if probe is None:
        return

    probe.custom_metric("generation_processed", total_processed)
    probe.custom_metric("generation_cache_hits", cache_hits)
    probe.custom_metric("generation_cache_misses", cache_misses)
    probe.custom_metric("generation_skipped_filtered", chats_skipped)
    probe.custom_metric("generation_model_elapsed_ms", model_elapsed_ms)
    if total_processed > 0:
        probe.custom_metric("generation_cache_hit_rate", cache_hits / total_processed)
        probe.custom_metric(
            "generation_cache_miss_rate", cache_misses / total_processed
        )


def _create_accumulator(
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
    tqdm_bar: standard_tqdm | None = None,
    tqdm_desc: str = "Calculating SHAP values",
    **generate_kwargs: dict[str, Any],
) -> tuple[int, list[tuple[Tensor, int, BaseMllmChat | None, ModelResponse]] | None]:
    """Generate model responses using a prep/infer pipeline.

    CPU-bound mask preparation (from_chat) runs in dedicated prep threads
    and feeds a bounded queue consumed by GPU/inference threads. This hides
    prep latency behind inference and keeps the GPU continuously busy.

    If the model overrides ``generate_batch``, a single GPU thread collects
    micro-batches for higher tensor-core utilization.
    """
    accumulator = _create_accumulator(
        masks=masks,
        responses=responses,
        verbose=verbose,
    )

    # Progress bar setup — track completions, not fetches
    owns_bar = False
    if tqdm_bar is None and progress_bar:
        tqdm_bar = standard_tqdm(desc=tqdm_desc, leave=False)
        owns_bar = True

    iterable_gen = enumerate(gen)

    # Shared state
    chats_skipped = 0
    cache_hits = 0
    cache_misses = 0
    model_elapsed_ms = 0.0
    error_flag = False
    iter_lock = Lock()
    stats_lock = Lock()

    # Detect if model has true batch support
    _supports_batch = type(model).generate_batch is not BaseMllmModel.generate_batch

    # Bounded queue connecting prep → GPU stages.
    # Each item is (i, mask, mask_hash, masked_chat | None, cached_response | None).
    _DONE = object()
    prefetch_queue: Queue = Queue(maxsize=n_generator_jobs * 2)

    # --- Prep workers: fetch masks, check cache, build masked chats (CPU) ---

    def prep_worker() -> None:
        nonlocal chats_skipped, error_flag

        while True:
            try:
                with iter_lock:
                    if error_flag:
                        return
                    try:
                        i, (mask, mask_hash) = next(iterable_gen)
                    except StopIteration:
                        return

                # Cache check (thread-safe by contract)
                if cache_manager.contains(mask_hash=mask_hash):
                    cached_response = cache_manager.extract(mask_hash=mask_hash)
                    prefetch_queue.put((i, mask, mask_hash, None, cached_response))
                else:
                    # CPU-intensive: construct masked chat
                    try:
                        masked_chat = type(source_chat).from_chat(
                            mask=mask, chat=source_chat
                        )
                    except AllTextTokensFilteredOutError:
                        with stats_lock:
                            chats_skipped += 1
                        if tqdm_bar is not None:
                            tqdm_bar.update(1)
                        continue
                    prefetch_queue.put((i, mask, mask_hash, masked_chat, None))

            except Exception as e:
                logger.error("Error in prep worker: %s", e, exc_info=True)
                with stats_lock:
                    error_flag = True
                return

    # --- GPU/inference worker (single-item mode) ---

    def gpu_worker() -> None:
        nonlocal cache_hits, cache_misses, model_elapsed_ms, error_flag

        while True:
            item = prefetch_queue.get()
            if item is _DONE:
                prefetch_queue.put(_DONE)
                return

            i, mask, mask_hash, masked_chat, cached_response = item

            if error_flag:
                continue

            if cached_response is not None:
                with stats_lock:
                    accumulator.append(
                        mask=mask,
                        mask_hash=mask_hash,
                        masked_chat=None,
                        model_response=cached_response,
                    )
                    cache_hits += 1
                    if tqdm_bar is not None:
                        tqdm_bar.update(1)
            else:
                try:
                    t0 = time()
                    probe = cache_manager._probe
                    if probe is None:
                        model_response = model.generate(
                            chat=masked_chat,
                            keep_history=verbose,
                            **generate_kwargs,
                        )
                    else:
                        with probe.timing("model"):
                            model_response = model.generate(
                                chat=masked_chat,
                                keep_history=verbose,
                                **generate_kwargs,
                            )
                    elapsed = (time() - t0) * 1000.0
                except Exception as e:
                    logger.error("Error in GPU worker: %s", e, exc_info=True)
                    with stats_lock:
                        error_flag = True
                    return

                with stats_lock:
                    accumulator.append(
                        mask=mask,
                        mask_hash=mask_hash,
                        masked_chat=masked_chat,
                        model_response=model_response,
                    )
                    cache_misses += 1
                    model_elapsed_ms += elapsed
                    if tqdm_bar is not None:
                        tqdm_bar.update(1)

                if not verbose:
                    del masked_chat
                    del model_response

    # --- GPU/inference worker (batched mode) ---

    def gpu_worker_batched() -> None:
        """Collect micro-batches from queue and call generate_batch."""
        nonlocal cache_hits, cache_misses, model_elapsed_ms, error_flag
        batch_size = n_generator_jobs

        while True:
            # Collect up to batch_size items needing inference
            batch_items: list[tuple] = []
            done = False

            while len(batch_items) < batch_size:
                # Block on first item, non-blocking drain for rest
                if not batch_items:
                    item = prefetch_queue.get()
                else:
                    try:
                        item = prefetch_queue.get_nowait()
                    except Empty:
                        break  # queue empty, process what we have

                if item is _DONE:
                    done = True
                    break

                if error_flag:
                    continue

                i, mask, mask_hash, masked_chat, cached_response = item

                if cached_response is not None:
                    # Cache hit — accumulate immediately
                    with stats_lock:
                        accumulator.append(
                            mask=mask,
                            mask_hash=mask_hash,
                            masked_chat=None,
                            model_response=cached_response,
                        )
                        cache_hits += 1
                        if tqdm_bar is not None:
                            tqdm_bar.update(1)
                else:
                    batch_items.append((i, mask, mask_hash, masked_chat))

            # Run batched inference
            if batch_items and not error_flag:
                chats_for_batch = [item[3] for item in batch_items]
                try:
                    t0 = time()
                    probe = cache_manager._probe
                    if probe is None:
                        batch_responses = model.generate_batch(
                            chats=chats_for_batch,
                            keep_history=verbose,
                            **generate_kwargs,
                        )
                    else:
                        with probe.timing("model"):
                            batch_responses = model.generate_batch(
                                chats=chats_for_batch,
                                keep_history=verbose,
                                **generate_kwargs,
                            )
                    elapsed = (time() - t0) * 1000.0
                except Exception as e:
                    logger.error("Error in batched GPU worker: %s", e, exc_info=True)
                    with stats_lock:
                        error_flag = True
                    return

                with stats_lock:
                    for (bi, b_mask, b_hash, b_chat), resp in zip(
                        batch_items, batch_responses
                    ):
                        accumulator.append(
                            mask=b_mask,
                            mask_hash=b_hash,
                            masked_chat=b_chat,
                            model_response=resp,
                        )
                    cache_misses += len(batch_items)
                    model_elapsed_ms += elapsed
                    if tqdm_bar is not None:
                        tqdm_bar.update(len(batch_items))

                if not verbose:
                    del chats_for_batch
                    del batch_responses

            if done:
                prefetch_queue.put(_DONE)
                return

    # Launch pipeline: prep threads feed GPU threads
    n_prep = min(2, n_generator_jobs)
    prep_threads = [Thread(target=prep_worker) for _ in range(n_prep)]

    if _supports_batch:
        # True batching: 1 GPU thread collects micro-batches
        gpu_threads = [Thread(target=gpu_worker_batched)]
    else:
        # Threaded parallelism: N workers with individual generate() calls
        gpu_threads = [Thread(target=gpu_worker) for _ in range(n_generator_jobs)]

    for t in prep_threads:
        t.start()
    for t in gpu_threads:
        t.start()

    # Wait for prep to exhaust all masks, then signal GPU workers
    for t in prep_threads:
        t.join()
    prefetch_queue.put(_DONE)

    for t in gpu_threads:
        t.join()

    if owns_bar and tqdm_bar is not None:
        tqdm_bar.close()

    if error_flag:
        raise WorkerExecutionError("Error occurred in SHAP explainer worker thread.")

    _report_generation_metrics(
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        chats_skipped=chats_skipped,
        model_elapsed_ms=model_elapsed_ms,
        cache_manager=cache_manager,
    )

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
    tqdm_bar: tqdm | None = None,
    tqdm_desc: str = "Calculating SHAP values",
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

    if tqdm_bar is not None:
        iterable_gen = enumerate(gen)
    else:
        iterable_gen = enumerate(
            tqdm(gen, desc=tqdm_desc, leave=False) if progress_bar else gen
        )

    chats_skipped = 0
    cache_hits = 0
    cache_misses = 0
    model_elapsed_ms = 0.0

    for i, (mask, mask_hash) in iterable_gen:
        try:
            t0 = time()
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
        if masked_chat is None:
            cache_hits += 1
        else:
            cache_misses += 1
            model_elapsed_ms += (time() - t0) * 1000.0
        if tqdm_bar is not None:
            tqdm_bar.update(1)

        if not verbose:
            # cleanup large refs; avoid forcing global GC in hot loop
            del masked_chat
            del model_response

    _report_generation_metrics(
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        chats_skipped=chats_skipped,
        model_elapsed_ms=model_elapsed_ms,
        cache_manager=cache_manager,
    )

    return chats_skipped, accumulator.history

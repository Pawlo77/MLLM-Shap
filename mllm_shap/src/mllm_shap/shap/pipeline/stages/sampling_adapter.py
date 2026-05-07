"""Shared sampling adapter for split generation paths."""

from logging import Logger
from typing import Any, Callable

import torch
from torch import Tensor

from ...base._generate_responses import generate_responses
from ...base._mask_generator import MaskGenerator
from ...base._masks_manager import MasksManager
from ...core.engine import SamplingEngine
from ...core.sampling import CallableAdapterStrategy
from ....connectors.base.chat import BaseMllmChat
from ....connectors.base.model import BaseMllmModel
from ....connectors.base.model_response import ModelResponse


def build_masks_generator(
    get_next_split: Callable[..., Tensor | None],
    get_num_splits: Callable[[int], int],
    mask_manager: MasksManager,
    device: torch.device,
    masks: list[Tensor],
    allow_mask_duplicates: bool,
    allow_full_or_empty: bool,
) -> MaskGenerator:
    """Build a mask generator using split callbacks."""
    strategy = CallableAdapterStrategy(
        get_next_split=get_next_split,
        get_num_splits=get_num_splits,
    )
    engine = SamplingEngine(
        strategy=strategy,
        allow_mask_duplicates=allow_mask_duplicates,
        allow_full_or_empty=allow_full_or_empty,
        probe=mask_manager._probe,
    )
    return engine.create_generator(
        mask_manager=mask_manager,
        device=device,
        masks=masks,
    )


def run_sampling_generation(
    get_next_split: Callable[..., Tensor | None],
    get_num_splits: Callable[[int], int],
    mask_manager: MasksManager,
    device: torch.device,
    masks: list[Tensor],
    allow_mask_duplicates: bool,
    allow_full_or_empty: bool,
    logger: Logger,
    tqdm_bar: Any | None = None,
    tqdm_desc: str = "Calculating SHAP values",
    responses: list[ModelResponse] | None = None,
    source_chat: BaseMllmChat | None = None,
    model: BaseMllmModel | None = None,
    generate_responses_fn: Callable[..., tuple[int, Any]] = generate_responses,
    get_masks_generator: Callable[
        [MasksManager, torch.device, list[Tensor], bool], MaskGenerator
    ]
    | None = None,
    **generate_kwargs: Any,
) -> tuple[
    tuple[int, list[tuple[Tensor, int, BaseMllmChat | None, ModelResponse]] | None],
    int,
]:
    """Run response generation and return result with generated mask count."""
    if get_masks_generator is not None:
        try:
            gen = get_masks_generator(
                mask_manager=mask_manager,
                device=device,
                masks=masks,
                allow_full_or_empty=allow_full_or_empty,
            )
        except TypeError:
            gen = get_masks_generator(
                mask_manager=mask_manager,
                device=device,
                masks=masks,
            )
    else:
        gen = build_masks_generator(
            get_next_split=get_next_split,
            get_num_splits=get_num_splits,
            mask_manager=mask_manager,
            device=device,
            masks=masks,
            allow_mask_duplicates=allow_mask_duplicates,
            allow_full_or_empty=allow_full_or_empty,
        )

    result = generate_responses_fn(
        masks=masks,
        responses=responses,
        source_chat=source_chat,
        model=model,
        gen=gen,
        tqdm_bar=tqdm_bar,
        tqdm_desc=tqdm_desc,
        **generate_kwargs,
    )

    stats = getattr(gen, "stats", None)
    if stats is not None:
        logger.info(
            "Sampling stats: candidates=%d yielded=%d skipped(full_or_empty)=%d skipped(invalid)=%d skipped(duplicates)=%d elapsed_ms=%.2f",
            stats.candidate_splits,
            stats.yielded_masks,
            stats.skipped_full_or_empty,
            stats.skipped_invalid_masks,
            stats.skipped_duplicates,
            stats.elapsed_ms,
        )
        probe = mask_manager._probe
        if probe is not None:
            probe.custom_metric("sampling_candidate_splits", stats.candidate_splits)
            probe.custom_metric("sampling_yielded_masks", stats.yielded_masks)
            probe.custom_metric(
                "sampling_skipped_full_or_empty", stats.skipped_full_or_empty
            )
            probe.custom_metric(
                "sampling_skipped_invalid_masks", stats.skipped_invalid_masks
            )
            probe.custom_metric("sampling_skipped_duplicates", stats.skipped_duplicates)
            probe.custom_metric("sampling_elapsed_ms", stats.elapsed_ms)
            if stats.candidate_splits > 0:
                probe.custom_metric(
                    "sampling_efficiency_ratio",
                    stats.yielded_masks / stats.candidate_splits,
                )
                probe.custom_metric(
                    "sampling_skip_rate",
                    (stats.candidate_splits - stats.yielded_masks)
                    / stats.candidate_splits,
                )

    return result, gen.generated_masks

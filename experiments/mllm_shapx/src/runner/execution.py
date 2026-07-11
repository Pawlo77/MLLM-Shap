"""Main orchestration for executing one expanded variant."""

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from mllm_shap.connectors.base.audio import SpectrogramGuidedAligner
from mllm_shap.shap.base._masks_manager import MasksManager
from tqdm.auto import tqdm

from ..config import ExperimentSet
from ..constants import audio_column_for, is_text_only_modality
from ..data import choose_prompt_text_column, extract_texts_from_row
from ..factory import (
    build_chat,
    build_explainer_for_variant,
    build_generation_kwargs,
    build_token_filter,
)
from ..mlflow_tracker import start_mlflow_run
from .io_utils import flatten_telemetry_metrics
from .stages import ChatBuilder, ExplainerRunner, ResultWriter, RowSelector, cleanup_gpu
from .types import ExpandedVariant
from .variants import pick_device

LOGGER = logging.getLogger(__name__)


def _prepare_sample(
    row: Dict[str, Any],
    text_col: str,
    audio_col_name: Any,
    cfg: ExperimentSet,
    chat_builder: ChatBuilder,
) -> Tuple[List[str], List[bytes] | None, Any, int]:
    """Prepare one sample: extract data and build chat (CPU-intensive).

    Returns:
        (user_texts, audio_bytes_list, chat, n_pre)
    """
    audio_bytes_list: List[bytes] | None = None
    if audio_col_name is not None:
        col_name = cfg.dataset.column_mapping.audio or audio_col_name.value
        if col_name in row:
            audio_val = row[col_name]
            if isinstance(audio_val, (list, np.ndarray)):
                audio_bytes_list = list(audio_val)
            else:
                audio_bytes_list = [audio_val]
        else:
            raise KeyError(
                f"Expected '{col_name}' in row for {cfg.modality.input_modality} input modality."
            )

    user_texts = extract_texts_from_row(row[text_col])
    chat = chat_builder.build(user_texts=user_texts, audio_bytes_list=audio_bytes_list)

    mask = getattr(chat, "shap_values_mask", None)
    if mask is not None and hasattr(mask, "sum"):
        n_pre = int(mask.sum().item())
    else:
        n_pre = int(MasksManager(chat).n)

    return user_texts, audio_bytes_list, chat, n_pre


def run_single_sentence_variant(
    cfg: ExperimentSet,
    run: ExpandedVariant,
    df: pd.DataFrame,
    resume: bool,
) -> None:
    """Execute one expanded variant across selected rows, storing results in MLflow."""
    device = pick_device(cfg.device)
    torch.manual_seed(cfg.selection.shuffle_seed or 42)

    effective_shap = cfg.get_effective_shap(run.variant)
    effective_gen = cfg.get_effective_generation(run.variant)
    effective_emb = cfg.get_effective_embedding(run.variant)

    spec: Dict[str, Any] = {
        "experiment_set_id": cfg.experiment_set_id,
        "run_slug": run.run_slug,
        "connector": cfg.connector,
        "modality": cfg.modality.model_dump(),
        "variant": {
            "explainer_type": run.variant.explainer_type,
            "num_samples": run.num_samples,
            "fraction": run.fraction,
            "linear": run.linear,
            "hierarchical": {
                "k": run.hier_k,
                "shap_type": run.hier_shap_type,
                "shap_num_samples": run.hier_shap_num_samples,
                "shap_fraction": run.hier_shap_fraction,
                "first_layer_type": run.hier_first_layer_type,
                "first_layer_num_samples": run.hier_first_layer_num_samples,
                "first_layer_fraction": run.hier_first_layer_fraction,
                "use_importance_sampling": True,
                "importance_min_fraction": run.hier_importance_min_fraction,
                "mode": run.hier_mode,
            },
        },
        "dataset": cfg.dataset.model_dump(),
        "selection": cfg.selection.model_dump(),
        "generation": effective_gen.model_dump(),
        "shap": effective_shap.model_dump(),
        "embedding": effective_emb.model_dump(),
        "runtime": cfg.runtime.model_dump(),
    }

    run_name = f"{cfg.experiment_set_id}__{run.run_slug}"
    tracker = start_mlflow_run(
        cfg.mlflow,
        run_name=run_name,
        experiment_set_id=cfg.experiment_set_id,
    )

    # Resume: query MLflow for previously completed samples
    completed_set: set[int] = set()
    if resume:
        completed_set = tracker.try_resume()
        if completed_set:
            LOGGER.info(
                "Resuming run '%s': %d samples already completed.",
                run.run_slug,
                len(completed_set),
            )

    explainer = build_explainer_for_variant(
        device,
        effective_shap,
        run.variant,
        cfg.connector,
        embedding_cfg=effective_emb,
        output_modality=cfg.modality.output_modality,
        connector_kwargs=cfg.connector_kwargs,
        concrete_num_samples=123 if run.linear else run.num_samples,
        concrete_fraction=run.fraction,
        hier_k=run.hier_k,
        hier_shap_type=run.hier_shap_type,
        hier_shap_num_samples=run.hier_shap_num_samples,
        hier_shap_fraction=run.hier_shap_fraction,
        hier_first_layer_type=run.hier_first_layer_type,
        hier_first_layer_num_samples=run.hier_first_layer_num_samples,
        hier_first_layer_fraction=run.hier_first_layer_fraction,
        hier_importance_min_fraction=run.hier_importance_min_fraction,
        hier_mode=run.hier_mode,
    )

    token_filter = build_token_filter(effective_shap.token_filter)
    audio_seg_method = cfg.audio_segmentation.method
    aligner = (
        SpectrogramGuidedAligner(
            device=torch.device(cfg.audio_segmentation.aligner_device)
        )
        if audio_seg_method == "sgpa"
        else None
    )

    text_col = choose_prompt_text_column(df, cfg.dataset.column_mapping.text)
    is_text_modality = is_text_only_modality(cfg.modality.input_modality)
    audio_col_name = audio_column_for(cfg.modality.input_modality)

    row_selector = RowSelector(cfg, df)
    chat_builder = ChatBuilder(
        model=explainer.model,
        input_modality=cfg.modality.input_modality,
        audio_segmentation_method=audio_seg_method,
        token_filter=token_filter,
        aligner=aligner,
        chat_cfg=cfg.chat,
        build_chat_fn=build_chat,
    )
    explainer_runner = ExplainerRunner(explainer, cfg.runtime)
    generation_kwargs = build_generation_kwargs(effective_gen)

    desired_total = cfg.selection.max_samples
    if desired_total is None:
        remaining_needed = float("inf")
    else:
        remaining_needed = max(int(desired_total) - len(completed_set), 0)

    progress_bar = (
        tqdm(
            total=int(remaining_needed), desc=f"Samples ({run.run_slug})", unit="sample"
        )
        if remaining_needed != float("inf") and remaining_needed > 0
        else None
    )

    with tracker:
        tracker.log_params_shallow(spec)
        tracker.log_dict(spec, artifact_file="summary/run_spec.json")
        result_writer = ResultWriter(tracker, cfg)

        matched_ctr = 0
        min_tokens = cfg.selection.min_prompt_tokens
        max_tokens = cfg.selection.max_prompt_tokens

        # Prefetch pool: prepares next sample's chat while GPU runs current one
        prep_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sample_prep")
        pending_prep: Future | None = None
        pending_row_idx: int | None = None

        # Use explicit iterator + lookahead buffer for prefetch peeking
        row_iter = iter(row_selector.iterate())
        lookahead: list[Tuple[int, Dict[str, Any]]] = []

        def _next_row() -> Tuple[int, Dict[str, Any]] | None:
            """Pop from lookahead buffer or advance the iterator."""
            if lookahead:
                return lookahead.pop(0)
            return next(row_iter, None)

        def _peek_next_eligible() -> Tuple[int, Dict[str, Any]] | None:
            """Peek ahead to find the next row not in completed_set."""
            item = next(row_iter, None)
            while item is not None:
                idx, _r = item
                if idx not in completed_set:
                    lookahead.append(item)
                    return item
                item = next(row_iter, None)
            return None

        try:
            item = _next_row()
            while item is not None:
                row_idx, row = item

                if matched_ctr >= remaining_needed:
                    break
                if row_idx in completed_set:
                    item = _next_row()
                    continue

                # Use prefetched result if available for this row, else prepare now
                if pending_prep is not None and pending_row_idx == row_idx:
                    user_texts, audio_bytes_list, chat, n_pre = pending_prep.result()
                    pending_prep = None
                    pending_row_idx = None
                else:
                    # Discard stale prefetch (row was skipped or mismatch)
                    if pending_prep is not None:
                        pending_prep.cancel()
                        pending_prep = None
                        pending_row_idx = None
                    user_texts, audio_bytes_list, chat, n_pre = _prepare_sample(
                        row, text_col, audio_col_name, cfg, chat_builder
                    )

                if is_text_modality:
                    if (min_tokens is not None and n_pre < int(min_tokens)) or (
                        max_tokens is not None and n_pre > int(max_tokens)
                    ):
                        item = _next_row()
                        continue

                if run.linear:
                    from .helpers import _LinearSampleScaler, _try_set_num_samples

                    scaled_num_samples = _LinearSampleScaler(factor=run.linear).scale(
                        n_pre=n_pre
                    )
                    if not _try_set_num_samples(
                        explainer=explainer, num_samples=scaled_num_samples
                    ):
                        explainer = build_explainer_for_variant(
                            device,
                            effective_shap,
                            run.variant,
                            cfg.connector,
                            embedding_cfg=effective_emb,
                            output_modality=cfg.modality.output_modality,
                            connector_kwargs=cfg.connector_kwargs,
                            concrete_num_samples=scaled_num_samples,
                            concrete_fraction=run.fraction,
                        )
                        explainer_runner = ExplainerRunner(explainer, cfg.runtime)

                # Submit prefetch for next eligible row before starting inference.
                # Overlaps next sample's CPU prep with current sample's GPU work.
                if pending_prep is None:
                    peeked = _peek_next_eligible()
                    if peeked is not None:
                        pending_row_idx = peeked[0]
                        pending_prep = prep_pool.submit(
                            _prepare_sample,
                            peeked[1],
                            text_col,
                            audio_col_name,
                            cfg,
                            chat_builder,
                        )

                result, runtime_sec, n_calls = explainer_runner.run(
                    chat, generation_kwargs
                )
                n_post = int(MasksManager(getattr(result, "base_chat", chat)).n)
                if n_post != n_pre:
                    LOGGER.warning(
                        "Token count mismatch for row=%d: pre=%d post=%d",
                        row_idx,
                        n_pre,
                        n_post,
                    )

                telemetry_data = getattr(result, "telemetry_data", None)
                telemetry_metrics = flatten_telemetry_metrics(telemetry_data)

                result_writer.save_sample(
                    row_idx=row_idx,
                    row=row,
                    result=result,
                    runtime_sec=runtime_sec,
                    n_calls=n_calls,
                    user_texts=user_texts,
                    input_modality=cfg.modality.input_modality,
                    output_modality=cfg.modality.output_modality,
                    audio_bytes_list=audio_bytes_list,
                    explainer=explainer,
                    run=run,
                    telemetry_metrics=telemetry_metrics,
                )

                completed_set.add(row_idx)
                matched_ctr += 1

                if progress_bar is not None:
                    progress_bar.update(1)

                cleanup_gpu(result, cfg.runtime, sample_idx=matched_ctr)
                item = _next_row()

            result_writer.save_aggregate(run.run_slug)
        finally:
            prep_pool.shutdown(wait=False)

    if progress_bar is not None:
        progress_bar.close()

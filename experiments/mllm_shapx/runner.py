"""Core orchestration: expand variants and execute runs."""

import gc
import json
import logging
import os
import time
from dataclasses import dataclass
from logging import Logger
from typing import Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd
import torch
from mllm_shap.connectors import ModelConfig
from mllm_shap.connectors.filters import ExcludePunctuationTokensFilter
from mllm_shap.shap.base._masks_manager import MasksManager
from mllm_shap.shap.base.approx import BaseShapApproximation
from mllm_shap.shap.hierarchical import HierarchicalExplainer
from mllm_shap.shap.neyman._base import BaseComplementaryNeymanShapExplainer

from .config import ExperimentSet, ExplainerVariant
from .data import (
    choose_prompt_text_column,
    extract_texts_from_row,
    iter_rows_for_selection,
)
from .constants import AudioCol, ExplainerType, InputModality
from .factory import build_chat, build_explainer_for_variant
from .serialization import compute_modality_summary, serialize_conversation
from .audio_utils import serialize_result_with_audio
from .storage import (
    existing_completed_from_disk,
    load_checkpoint,
    make_run_dir,
    save_json,
    update_checkpoint,
)
from .wandb_utils import (
    log_metrics,
    wandb_init_if_enabled,
    wandb_log_artifact,
    wandb_log_dir_incremental,
    log_audio_artifacts,
)

LOGGER: Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _LinearSampleScaler:
    """Policy for deriving linear ``num_samples`` from token count.

    Attributes:
        factor: Multiplier used in ``factor * n_pre^2``.
    """

    factor: float

    def scale(self, n_pre: int) -> int:
        """Compute even ``num_samples`` value.

        Args:
            n_pre: Number of explainable tokens before generation.

        Returns:
            Even sampling budget used by complementary explainers.
        """
        scaled_num_samples = int(self.factor * n_pre * n_pre)
        return (
            scaled_num_samples
            if scaled_num_samples % 2 == 0
            else scaled_num_samples + 1
        )


def _try_set_num_samples(explainer: Any, num_samples: int) -> bool:
    """
    Update sampling budget in-place for approximation explainers.

    Args:
        explainer: High-level explainer object.
        num_samples: New sampling budget.

    Returns:
        True if budget updated in-place, False if caller should rebuild explainer.
    """
    shap_explainer = getattr(explainer, "shap_explainer", None)
    if isinstance(shap_explainer, BaseShapApproximation):
        shap_explainer.num_samples = int(num_samples)
        return True
    return False


@dataclass(frozen=True)
class ExpandedVariant:
    """A concrete materialization of a user-declared variant."""

    run_slug: str
    variant: ExplainerVariant
    fraction: Optional[float]
    num_samples: Optional[int]
    linear: Optional[float]
    # hierarchical-specific
    hier_k: Optional[int] = None
    hier_shap_type: Optional[str] = None
    hier_shap_num_samples: Optional[int] = None
    hier_shap_fraction: Optional[float] = None
    hier_first_layer_type: Optional[str] = None
    hier_first_layer_num_samples: Optional[int] = None
    hier_first_layer_fraction: Optional[float] = None
    hier_importance_min_fraction: Optional[float] = None


def pick_device(name: Optional[str]) -> torch.device:
    """Resolve the torch device preference."""
    if name is None:
        return (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    return torch.device(name)


def expand_variants(
    cfg: ExperimentSet,
) -> List[ExpandedVariant]:
    """
    Materialize user-defined variants into concrete runs.
    - exact: single run
    - limited_mc/standard_mc/complementary:
        - num_samples: one run per entry
        - fractions: one run per entry
    - neyman:
        - single AUTO run (ignores ns/fractions even if provided)
    - hierarchical:
        - unchanged here (will be tightened later to use complementary/neyman only)
    """
    out: List[ExpandedVariant] = []

    for v in cfg.experiments:
        t = v.explainer_type.lower()

        if t == ExplainerType.EXACT.value:
            slug = v.name or ExplainerType.EXACT.value
            out.append(
                ExpandedVariant(
                    run_slug=slug,
                    variant=v,
                    fraction=None,
                    num_samples=None,
                    linear=None,
                )
            )
            continue

        if t in (
            ExplainerType.LIMITED_MC.value,
            ExplainerType.STANDARD_MC.value,
            ExplainerType.LIMITED_CC.value,
            ExplainerType.STANDARD_CC.value,
            ExplainerType.LIMITED_NEYMAN.value,
            ExplainerType.STANDARD_NEYMAN.value,
        ):
            if v.num_samples:
                for ns in v.num_samples:
                    ns_i = int(ns)
                    suffix = f"{t}_ns{ns_i}"
                    slug = (v.name + "_" + suffix) if v.name else suffix
                    out.append(ExpandedVariant(slug, v, None, ns_i, None))
            if v.fractions:
                for frac in v.fractions:
                    f = float(frac)
                    suffix = f"{t}_frac{str(f).replace('.', '_')}"
                    slug = (v.name + "_" + suffix) if v.name else suffix
                    out.append(ExpandedVariant(slug, v, f, None, None))
            if v.linear:
                for lin in v.linear:
                    lin = float(lin)
                    suffix = f"{t}_lin{str(lin).replace('.', '_')}"
                    slug = (v.name + "_" + suffix) if v.name else suffix
                    out.append(ExpandedVariant(slug, v, None, None, lin))
            continue

        if t == ExplainerType.HIERARCHICAL.value:
            none_str = "none"
            ks = v.hierarchical_ks or []
            if not ks:
                ks = [10]  # safe default

            inner_type = (v.hierarchical_shap_type or "limited_neyman").lower()
            # inner param sweeps (exclusive-or style; if none provided, produce one AUTO run)
            inner_ns = v.hierarchical_shap_num_samples or []
            inner_fracs = v.hierarchical_shap_fractions or []

            # first-layer choices
            fl_type = (v.hierarchical_first_layer_type or none_str).lower()
            fl_ns = v.hierarchical_first_layer_num_samples or []
            fl_fracs = v.hierarchical_first_layer_fractions or []

            imp_min_fracs = v.hierarchical_importance_min_fractions or [0.1]

            # normalize empty lists to [None] for cartesian product
            def _nz(x: list[Any] | None) -> list[Any]:
                """Return input list or [None] if empty."""
                return x if x else [None]

            for k in ks:
                for impmf in imp_min_fracs:
                    for inn_ns in _nz(inner_ns):
                        for inn_fr in _nz(inner_fracs):
                            # if both None => AUTO for inner
                            # now first-layer combos
                            if fl_type == none_str:
                                slug = f"hier_{inner_type}_k{k}_imp{str(impmf).replace('.', '_')}"
                                if inn_ns is not None:
                                    slug += f"_ns{inn_ns}"
                                if inn_fr is not None:
                                    slug += f"_frac{str(inn_fr).replace('.', '_')}"
                                out.append(
                                    ExpandedVariant(
                                        run_slug=(v.name + "_" + slug)
                                        if v.name
                                        else slug,
                                        variant=v,
                                        fraction=None,
                                        num_samples=None,
                                        linear=None,
                                        hier_k=int(k),
                                        hier_shap_type=inner_type,
                                        hier_shap_num_samples=int(inn_ns)
                                        if inn_ns is not None
                                        else None,
                                        hier_shap_fraction=float(inn_fr)
                                        if inn_fr is not None
                                        else None,
                                        hier_first_layer_type=none_str,
                                        hier_importance_min_fraction=float(impmf),
                                    )
                                )
                            else:
                                for flns in _nz(fl_ns):
                                    for flfr in _nz(fl_fracs):
                                        slug = f"hier_{inner_type}_k{k}_imp{str(impmf).replace('.', '_')}_fl{fl_type}"
                                        if inn_ns is not None:
                                            slug += f"_ns{inn_ns}"
                                        if inn_fr is not None:
                                            slug += (
                                                f"_frac{str(inn_fr).replace('.', '_')}"
                                            )
                                        if flns is not None:
                                            slug += f"_flns{flns}"
                                        if flfr is not None:
                                            slug += (
                                                f"_flfrac{str(flfr).replace('.', '_')}"
                                            )
                                        out.append(
                                            ExpandedVariant(
                                                run_slug=(v.name + "_" + slug)
                                                if v.name
                                                else slug,
                                                variant=v,
                                                fraction=None,
                                                num_samples=None,
                                                linear=None,
                                                hier_k=int(k),
                                                hier_shap_type=inner_type,
                                                hier_shap_num_samples=int(inn_ns)
                                                if inn_ns is not None
                                                else None,
                                                hier_shap_fraction=float(inn_fr)
                                                if inn_fr is not None
                                                else None,
                                                hier_first_layer_type=fl_type,
                                                hier_first_layer_num_samples=int(flns)
                                                if flns is not None
                                                else None,
                                                hier_first_layer_fraction=float(flfr)
                                                if flfr is not None
                                                else None,
                                                hier_importance_min_fraction=float(
                                                    impmf
                                                ),
                                            )
                                        )
            continue

        raise ValueError(f"Unsupported explainer_type: {v.explainer_type}")

    return out


def run_single_sentence_variant(
    cfg: ExperimentSet,
    run: ExpandedVariant,
    df: pd.DataFrame,
    resume: bool,
) -> None:
    """Execute one concrete variant over the selected rows and persist all artifacts."""
    device = pick_device(cfg.device)
    run_dir = make_run_dir(cfg.output_root, cfg.experiment_set_id, run.run_slug)
    torch.manual_seed(cfg.selection.shuffle_seed or 42)

    # ---- spec (saved once per variant)
    spec: Dict[str, Any] = {
        "experiment_set_id": cfg.experiment_set_id,
        "run_slug": run.run_slug,
        "connector": cfg.connector,
        "modality": {
            "input_modality": cfg.modality.input_modality,
            "output_modality": cfg.modality.output_modality,
        },
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
            },
        },
        "dataset": cfg.dataset.__dict__,
        "selection": cfg.selection.__dict__,
        "generation": cfg.generation.__dict__,
        "embedding": cfg.embedding.__dict__ if cfg.embedding else None,
        "shap": cfg.shap.__dict__,
        "device": str(device),
    }
    save_json(run_dir / "spec.json", spec)

    # ---- checkpoint
    ckpt_path = run_dir / "checkpoint.json"
    disable_checkpoint = os.environ.get("MLLM_SHAPX_DISABLE_CHECKPOINT") == "1"
    ckpt = {
        "completed_indices": [],
        "next_index": 0,
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    if not disable_checkpoint:
        ckpt = load_checkpoint(ckpt_path)

    if disable_checkpoint:
        LOGGER.info("Checkpointing disabled for run '%s'.", run.run_slug)
    elif resume:
        already = existing_completed_from_disk(run_dir)
        ckpt["completed_indices"] = sorted(
            set(ckpt["completed_indices"]).union(already)
        )
        ckpt.setdefault("next_index", 0)
        LOGGER.info(
            "Resuming run '%s': %d completed samples found on disk.",
            run.run_slug,
            len(ckpt["completed_indices"]),
        )
    else:
        ckpt = {
            "completed_indices": [],
            "next_index": 0,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        save_json(ckpt_path, ckpt)

    completed_set = set(ckpt["completed_indices"])

    # ---- W&B
    run_config = {**spec, "python_version": f"{torch.__version__}"}
    wb_run = wandb_init_if_enabled(
        cfg.wandb,
        run_name=f"{cfg.experiment_set_id}__{run.run_slug}",
        run_config=run_config,
    )

    # ---- modality settings
    input_modality = cfg.modality.get_input_modality()
    output_modality = cfg.modality.get_output_modality()

    # ---- explainer
    explainer = build_explainer_for_variant(
        device,
        cfg.shap,
        run.variant,
        cfg.connector,
        embedding_cfg=cfg.embedding,
        output_modality=output_modality,
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
    )

    # ---- selection policy: scan ALL rows after start_index (maybe shuffled), break after matched == target
    text_col = choose_prompt_text_column(df)
    is_text_only = input_modality == InputModality.TEXT
    token_filter = ExcludePunctuationTokensFilter()

    min_t = cfg.selection.min_prompt_tokens
    max_t = cfg.selection.max_prompt_tokens

    # How many NEW matched samples do we still need (respect resume)?
    desired_total = cfg.selection.max_samples
    if desired_total is None:
        remaining_needed = float("inf")
    else:
        # Cap to non-negative in case completed already >= desired_total
        remaining_needed = max(int(desired_total) - len(completed_set), 0)

    if remaining_needed == 0:
        LOGGER.info(
            "Target of max_samples=%s already satisfied by existing samples. Nothing to do.",
            str(cfg.selection.max_samples),
        )

    # Counters for logging
    scanned_ctr = 0
    ge_min_ctr = 0
    matched_ctr = 0  # rows that passed bounds and we actually processed now (new)

    # Iterate ALL rows (post start_index, optional shuffle). max_samples=None -> full scan.
    for row_idx, row in iter_rows_for_selection(
        df=df,
        start_index=cfg.selection.start_index,
        max_samples=None,  # IMPORTANT: full scan; we will break on matched count
        shuffle_seed=cfg.selection.shuffle_seed,
    ):
        # If we've collected enough, stop.
        if matched_ctr >= remaining_needed:
            break

        # Skip if already processed (resume)
        if row_idx in completed_set:
            continue

        scanned_ctr += 1

        # Audio column resolution based on input modality
        audio_bytes_list: Optional[list[bytes]] = None

        # Determine which audio column to use based on modality
        needs_male_audio = input_modality in (
            InputModality.AUDIO_MALE,
            InputModality.INTERLEAVED_TEXT_FIRST_MALE,
            InputModality.INTERLEAVED_AUDIO_FIRST_MALE,
        )
        needs_female_audio = input_modality in (
            InputModality.AUDIO_FEMALE,
            InputModality.INTERLEAVED_TEXT_FIRST_FEMALE,
            InputModality.INTERLEAVED_AUDIO_FIRST_FEMALE,
        )

        if needs_male_audio:
            if AudioCol.MALE.value in row:
                audio_bytes_list = row[AudioCol.MALE.value]
                if not isinstance(audio_bytes_list, (list, np.ndarray)):
                    audio_bytes_list = [audio_bytes_list]
            else:
                raise KeyError(
                    f"Expected '{AudioCol.MALE.value}' in row for {input_modality} input modality."
                )
        elif needs_female_audio:
            if AudioCol.FEMALE.value in row:
                audio_bytes_list = row[AudioCol.FEMALE.value]
                if not isinstance(audio_bytes_list, (list, np.ndarray)):
                    audio_bytes_list = [audio_bytes_list]
            else:
                raise KeyError(
                    f"Expected '{AudioCol.FEMALE.value}' in row for {input_modality} input modality."
                )

        # Prompt resolution - returns list of texts for multi-turn support
        user_texts = extract_texts_from_row(row[text_col])

        # Build the SAME chat that will be passed to the explainer
        chat = build_chat(
            explainer.model,
            user_texts=user_texts,
            audio_bytes_list=audio_bytes_list,
            input_modality=input_modality,
            token_filter=token_filter,
        )

        # Ensure masks/tokens ready
        if hasattr(chat, "refresh"):
            chat.refresh(full=True)

        # Explainable-token counts
        mask = getattr(chat, "shap_values_mask", None)
        if mask is not None and hasattr(mask, "sum"):
            mask_sum = int(mask.sum().item())
            n_pre = mask_sum
        else:
            n_pre = int(MasksManager(chat).n)
            mask_sum = n_pre

        input_tok = int(getattr(chat, "input_tokens_num", n_pre))
        LOGGER.info(
            "At filter: n=%d | mask_sum=%d | input_tokens=%d | filter=%s",
            n_pre,
            mask_sum,
            input_tok,
            type(token_filter).__name__ if token_filter else "None",
        )

        # Count rows that satisfy min bound (for logging parity)
        if min_t is None or n_pre >= int(min_t):
            ge_min_ctr += 1

        # Apply bounds (only meaningful for text-only)
        in_bounds = True
        if is_text_only:
            if (min_t is not None and n_pre < int(min_t)) or (
                max_t is not None and n_pre > int(max_t)
            ):
                in_bounds = False

        if not in_bounds:
            continue  # not matched; keep scanning

        # ---- matched row: run explainer
        LOGGER.info("Running sample index %d for variant '%s'.", row_idx, run.run_slug)
        LOGGER.info(" | ".join(user_texts))

        if run.linear:
            scaled_num_samples = _LinearSampleScaler(factor=run.linear).scale(
                n_pre=n_pre
            )
            LOGGER.info(
                "Using linear explainer with scaled num_samples=%d.", scaled_num_samples
            )
            if not _try_set_num_samples(
                explainer=explainer, num_samples=scaled_num_samples
            ):
                # Fallback for explainers that do not expose mutable sampling budget.
                explainer = build_explainer_for_variant(
                    device,
                    cfg.shap,
                    run.variant,
                    cfg.connector,
                    embedding_cfg=cfg.embedding,
                    output_modality=output_modality,
                    concrete_num_samples=scaled_num_samples,
                    concrete_fraction=run.fraction,
                )

        generation_kwargs = {
            "max_new_tokens": int(cfg.generation.max_new_tokens),
            "model_config": ModelConfig(
                text_temperature=float(cfg.generation.text_temperature)
            ),
        }

        t0 = time.time()
        result = explainer(
            chat=chat,
            verbose=True,
            generation_kwargs=generation_kwargs,
            progress_bar=True,
        )
        runtime_sec = time.time() - t0
        n_calls = explainer.total_n_calls

        # Parity check
        n_post = int(MasksManager(getattr(result, "base_chat", chat)).n)
        if n_post != n_pre:
            LOGGER.warning(
                "Explainable-token drift detected: pre=%d post=%d (row=%d).",
                n_pre,
                n_post,
                row_idx,
            )

        # ---- serialize + metrics
        conv = result.full_chat.get_conversation()
        modality_summary = compute_modality_summary(conv)
        conv_json = serialize_conversation(conv)

        sample_result: Dict[str, Any] = {
            "row_index": int(row_idx),
            "language": row.get("language", "unknown"),
            "original_language": row.get("original_language", "unknown"),
            "runtime_sec": float(runtime_sec),
            "n_calls": n_calls,
            "prompt_texts": user_texts,  # List of texts for multi-turn
            "input_modality": input_modality.value,
            "output_modality": output_modality.value,
            "attr_summary": modality_summary,
            "conversation": conv_json,
        }

        # ---- audio artifacts
        audio_artifacts_dir = run_dir / "audio" / f"sample_{row_idx:05d}"
        audio_info = serialize_result_with_audio(
            result=result,
            output_dir=audio_artifacts_dir,
            input_audio_bytes=audio_bytes_list,
            input_modality=input_modality,
            output_modality=output_modality,
            sample_id=f"sample_{row_idx:05d}",
        )
        if audio_info.get("input_audio") or audio_info.get("output_audio"):
            sample_result["audio_artifacts"] = audio_info
            # Log audio to WandB
            if wb_run is not None:
                log_audio_artifacts(
                    wb_run,
                    audio_artifacts_dir,
                    input_modality,
                    output_modality,
                    f"sample_{row_idx:05d}",
                )

        if run.variant.explainer_type == ExplainerType.HIERARCHICAL.value:
            sample_result["num_levels"] = cast(HierarchicalExplainer, explainer).n_calls
        elif run.variant.explainer_type in (
            ExplainerType.LIMITED_NEYMAN.value,
            ExplainerType.STANDARD_NEYMAN.value,
        ):
            neyman_explainer = cast(
                BaseComplementaryNeymanShapExplainer, explainer.shap_explainer
            )
            # Access step count from neyman explainer

            sample_result["neyman_steps"] = (
                neyman_explainer._BaseComplementaryNeymanShapExplainer__step
            )

        sample_path = run_dir / "samples" / f"sample_{row_idx:05d}_result.json"
        save_json(sample_path, sample_result)
        if wb_run is not None:
            wandb_log_dir_incremental(
                wb_run,
                run_dir / "samples",
                artifact_name=f"{cfg.experiment_set_id}__{run.run_slug}-samples",
                artifact_type="samples",
                metadata={"subset": cfg.dataset.subset, "split": cfg.dataset.split},
            )

        log_metrics(
            wb_run,
            {
                "progress/sample_index": row_idx,
                "timing/runtime_sec": runtime_sec,
                "attr/abs_sum_text": modality_summary["abs_sum_text"],
                "attr/abs_sum_audio": modality_summary["abs_sum_audio"],
                "attr/frac_text": modality_summary["frac_text"],
                "attr/frac_audio": modality_summary["frac_audio"],
                "counts/text_tokens": modality_summary["count_text_tokens"],
                "counts/audio_segments": modality_summary["count_audio_segments"],
                "meta/language": row.get("language", "unknown"),
                "meta/input_modality": input_modality.value,
                "meta/output_modality": output_modality.value,
            },
        )

        # checkpoint after each sample
        if not disable_checkpoint:
            update_checkpoint(
                ckpt_path, ckpt, just_completed=row_idx, next_index=row_idx + 1
            )
        completed_set.add(row_idx)
        matched_ctr += 1

        # ===== Aggressive GPU cleanup after each sample =====
        # Clear the cache from the chat
        if hasattr(result, "full_chat") and result.full_chat.cache is not None:
            result.full_chat.cache = None

        # Clear history explicitly
        if result.history is not None:
            for mask, mask_hash, masked_chat, response in result.history:
                del mask, mask_hash, masked_chat, response
            del result.history

        # Delete result object
        del result

        # Force garbage collection
        gc.collect()

        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # ---- aggregate summary
    summaries: List[Dict[str, Any]] = []
    for p in sorted((run_dir / "samples").glob("sample_*_result.json")):
        with open(p, "r", encoding="utf-8") as f:
            summaries.append(json.load(f))

    if summaries:
        runtimes = [s["runtime_sec"] for s in summaries]
        frac_text = [s["attr_summary"]["frac_text"] for s in summaries]
        frac_audio = [s["attr_summary"]["frac_audio"] for s in summaries]
        agg = {
            "n_samples": len(summaries),
            "runtime_sec_total": float(sum(runtimes)),
            "runtime_sec_avg": float(np.mean(runtimes)),
            "avg_frac_text": float(np.mean(frac_text)),
            "avg_frac_audio": float(np.mean(frac_audio)),
        }
    else:
        agg = {
            "n_samples": 0,
            "runtime_sec_total": 0.0,
            "runtime_sec_avg": 0.0,
            "avg_frac_text": 0.0,
            "avg_frac_audio": 0.0,
        }

    save_json(run_dir / "summary" / "aggregate_metrics.json", agg)

    # ---- filter logs mirroring your earlier phrasing (with early-break denominators)
    if is_text_only and (min_t is not None):
        LOGGER.info(
            "Explainable-token filter: >= %d tokens matched %d / %d rows.",
            int(min_t),
            ge_min_ctr,
            scanned_ctr,
        )
    if is_text_only and (max_t is not None):
        # Matched both bounds; denominator = rows that met the min bound
        denom = ge_min_ctr if min_t is not None else scanned_ctr
        LOGGER.info(
            "Explainable-token filter: <= %d tokens matched %d / %d rows.",
            int(max_t),
            matched_ctr,
            denom,
        )

    # ---- W&B artifact
    if wb_run is not None:
        artifact_path = run_dir / "summary" / "aggregate_metrics.json"
        wandb_log_artifact(
            wb_run,
            artifact_path,
            artifact_name=f"{cfg.experiment_set_id}__{run.run_slug}-summary",
            artifact_type="summary",
            metadata={"subset": cfg.dataset.subset, "split": cfg.dataset.split},
        )
        wb_run.finish()

"""Core orchestration: expand variants and execute runs."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from mllm_shap.connectors import ModelConfig
from mllm_shap.shap.base._masks_manager import MasksManager
from mllm_shap.connectors.filters import ExcludePunctuationTokensFilter

from .config import ExperimentSet, ExplainerVariant
from .data import (
    choose_prompt_text_column,
    iter_rows_for_selection,
)
from .constants import AudioCol, ConnectorType, ExplainerType
from .factory import build_chat, build_explainer_for_variant
from .serialization import compute_modality_summary, serialize_conversation
from .storage import (
    existing_completed_from_disk,
    load_checkpoint,
    make_run_dir,
    save_json,
    update_checkpoint,
)
from .wandb_utils import log_metrics, wandb_init_if_enabled, wandb_log_artifact, wandb_log_dir_incremental

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExpandedVariant:
    """A concrete materialization of a user-declared variant."""

    run_slug: str
    variant: ExplainerVariant
    fraction: Optional[float]
    num_samples: Optional[int]
    base_variant: Optional[str] = None  # for hierarchical transparency


def pick_device(name: Optional[str]) -> torch.device:
    """Resolve the torch device preference."""
    if name is None:
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(name)


def expand_variants(cfg: ExperimentSet) -> List[ExpandedVariant]:  # pylint: disable=too-many-branches
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
            out.append(ExpandedVariant(run_slug=slug, variant=v, fraction=None, num_samples=None))
            continue

        if t in (ExplainerType.LIMITED_MC.value,
                 ExplainerType.STANDARD_MC.value,
                 ExplainerType.COMPLEMENTARY.value,
                 ExplainerType.NEYMAN.value):
            if v.num_samples:
                for ns in v.num_samples:
                    ns_i = int(ns)
                    suffix = f"{t}_ns{ns_i}"
                    slug = (v.name + "_" + suffix) if v.name else suffix
                    out.append(ExpandedVariant(slug, v, None, ns_i))
            if v.fractions:
                for frac in v.fractions:
                    f = float(frac)
                    suffix = f"{t}_frac{str(f).replace('.', '_')}"
                    slug = (v.name + "_" + suffix) if v.name else suffix
                    out.append(ExpandedVariant(slug, v, f, None))
            continue

        if t == ExplainerType.HIERARCHICAL.value:
            base = (
                v.hierarchical_base
                or (ExplainerType.LIMITED_MC.value if (v.num_samples or v.fractions) else ExplainerType.EXACT.value)
            ).lower()

            if base == ExplainerType.EXACT.value:
                slug = v.name or "hier_exact"
                out.append(ExpandedVariant(slug, v, None, None, base_variant=base))

            elif base in (
                ExplainerType.LIMITED_MC.value,
                ExplainerType.STANDARD_MC.value,
                ExplainerType.COMPLEMENTARY.value,
                ExplainerType.NEYMAN.value
            ):
                if v.num_samples:
                    for ns in v.num_samples:
                        ns_i = int(ns)
                        suffix = f"hier_{base}_ns{ns_i}"
                        slug = (v.name + "_" + suffix) if v.name else suffix
                        out.append(ExpandedVariant(slug, v, None, ns_i, base_variant=base))
                if v.fractions:
                    for frac in v.fractions:
                        f = float(frac)
                        suffix = f"hier_{base}_frac{str(f).replace('.', '_')}"
                        slug = (v.name + "_" + suffix) if v.name else suffix
                        out.append(ExpandedVariant(slug, v, f, None, base_variant=base))
                if not (v.num_samples or v.fractions):
                    slug = v.name or f"hier_{base}"
                    out.append(ExpandedVariant(slug, v, None, None, base_variant=base))
            else:
                raise ValueError(f"Unsupported hierarchical base: {base}")
            continue

        raise ValueError(f"Unsupported explainer_type: {v.explainer_type}")

    return out


def run_single_sentence_variant(  # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    cfg: ExperimentSet,
    run: ExpandedVariant,
    df: pd.DataFrame,
    resume: bool,
) -> None:
    """Execute one concrete variant over the selected rows and persist all artifacts."""
    device = pick_device(cfg.device)
    run_dir = make_run_dir(cfg.output_root, cfg.experiment_set_id, run.run_slug)

    # ---- spec (saved once per variant)
    spec: Dict[str, Any] = {
        "experiment_set_id": cfg.experiment_set_id,
        "run_slug": run.run_slug,
        "connector": cfg.connector,
        "variant": {
            "explainer_type": run.variant.explainer_type,
            "num_samples": run.num_samples,
            "fraction": run.fraction,
            "hierarchical_k": run.variant.hierarchical_k,
            "hierarchical_base": run.base_variant or run.variant.hierarchical_base,
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
    ckpt = load_checkpoint(ckpt_path)

    if resume:
        already = existing_completed_from_disk(run_dir)
        ckpt["completed_indices"] = sorted(set(ckpt["completed_indices"]).union(already))
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

    # ---- explainer
    explainer = build_explainer_for_variant(
        device,
        cfg.shap,
        run.variant,
        cfg.connector,
        embedding_cfg=cfg.embedding,
        concrete_num_samples=run.num_samples,
        concrete_fraction=run.fraction,
    )

    # ---- selection policy: scan ALL rows after start_index (maybe shuffled), break after matched == target
    text_col = choose_prompt_text_column(df)
    is_text_only = cfg.connector == ConnectorType.TRANSFORMERS_TEXT.value
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

        # Audio column resolution
        audio_bytes: Optional[bytes] = None
        if not is_text_only:
            if AudioCol.MALE.value in row:
                audio_bytes = row[AudioCol.MALE.value][0]
            elif AudioCol.FEMALE.value in row:
                audio_bytes = row[AudioCol.FEMALE.value][0]
            else:
                raise KeyError("Expected 'audio__male' or 'audio__female' in row for audio model.")

        # Prompt resolution
        v = row[text_col]
        user_text = v[0] if isinstance(v, list) and v else (str(v) if v is not None else "")

        # Build the SAME chat that will be passed to the explainer
        chat = build_chat(
            explainer.model,
            user_text=user_text,
            audio_bytes=audio_bytes,
            text_only=is_text_only,
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
            n_pre, mask_sum, input_tok, type(token_filter).__name__ if token_filter else "None",
        )

        # Count rows that satisfy min bound (for logging parity)
        if min_t is None or n_pre >= int(min_t):
            ge_min_ctr += 1

        # Apply bounds (only meaningful for text-only)
        in_bounds = True
        if is_text_only:
            if (min_t is not None and n_pre < int(min_t)) or (max_t is not None and n_pre > int(max_t)):
                in_bounds = False

        if not in_bounds:
            continue  # not matched; keep scanning

        # ---- matched row: run explainer
        LOGGER.info("Running sample index %d for variant '%s'.", row_idx, run.run_slug)
        LOGGER.info(user_text)

        generation_kwargs = {
            "max_new_tokens": int(cfg.generation.max_new_tokens),
            "model_config": ModelConfig(text_temperature=float(cfg.generation.text_temperature)),
        }

        t0 = time.time()
        result = explainer(
            chat=chat,
            verbose=True,
            generation_kwargs=generation_kwargs,
            progress_bar=True,
        )
        runtime_sec = time.time() - t0

        # Parity check
        n_post = int(MasksManager(getattr(result, "base_chat", chat)).n)
        if n_post != n_pre:
            LOGGER.warning(
                "Explainable-token drift detected: pre=%d post=%d (row=%d).",
                n_pre, n_post, row_idx
            )

        # ---- serialize + metrics
        conv = result.full_chat.get_conversation()
        modality = compute_modality_summary(conv)
        conv_json = serialize_conversation(conv)

        sample_result = {
            "row_index": int(row_idx),
            "language": row.get("language", "unknown"),
            "original_language": row.get("original_language", "unknown"),
            "runtime_sec": float(runtime_sec),
            "prompt_text": user_text,
            "attr_summary": modality,
            "conversation": conv_json,
        }

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
                "attr/abs_sum_text": modality["abs_sum_text"],
                "attr/abs_sum_audio": modality["abs_sum_audio"],
                "attr/frac_text": modality["frac_text"],
                "attr/frac_audio": modality["frac_audio"],
                "counts/text_tokens": modality["count_text_tokens"],
                "counts/audio_segments": modality["count_audio_segments"],
                "meta/language": row.get("language", "unknown"),
            },
        )

        # checkpoint after each sample
        update_checkpoint(ckpt_path, ckpt, just_completed=row_idx, next_index=row_idx + 1)
        completed_set.add(row_idx)
        matched_ctr += 1

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
            int(min_t), ge_min_ctr, scanned_ctr
        )
    if is_text_only and (max_t is not None):
        # Matched both bounds; denominator = rows that met the min bound
        denom = ge_min_ctr if min_t is not None else scanned_ctr
        LOGGER.info(
            "Explainable-token filter: <= %d tokens matched %d / %d rows.",
            int(max_t), matched_ctr, denom
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

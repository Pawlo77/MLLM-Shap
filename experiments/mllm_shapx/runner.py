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
from mllm_shap.shap import Explainer, McShapExplainer
from mllm_shap.shap.enums import Mode
from mllm_shap.shap.similarity import CosineSimilarity

from .config import (
    NORMALIZER_MAP,
    REDUCER_MAP,
    ExperimentSet,
    ExplainerVariant,
    ShapConfig,
)
from .data import (choose_prompt_text_column,
                   iter_rows_for_selection,
                   filter_df_by_max_prompt_tokens,
                   get_hf_text_tokenizer)
from .constants import AudioCol, ExplainerType, ModelKind
from .factory import build_chat, build_explainer_for_variant
from .serialization import compute_modality_summary, serialize_conversation
from .storage import (
    existing_completed_from_disk,
    load_checkpoint,
    make_run_dir,
    save_json,
    update_checkpoint,
)
from .wandb_utils import log_metrics, wandb_init_if_enabled, wandb_log_artifact

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExpandedVariant:
    """A concrete materialization of a user-declared variant."""
    run_slug: str
    variant: ExplainerVariant
    fraction: Optional[float]
    num_samples: Optional[int]


def pick_device(name: Optional[str]) -> torch.device:
    """Resolve the torch device preference."""
    if name is None:
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(name)


def expand_variants(cfg: ExperimentSet) -> List[ExpandedVariant]:
    """
    Materialize user-defined variants into concrete runs.
    - exact: one run
    - mc:
        - num_samples: one run per entry
        - fractions:   one run per entry
    """
    out: List[ExpandedVariant] = []
    for v in cfg.experiments:
        t = v.explainer_type.lower()
        if t == ExplainerType.EXACT.value:
            slug = v.name or ExplainerType.EXACT.value
            out.append(ExpandedVariant(run_slug=slug, variant=v, fraction=None, num_samples=None))
        else:
            if v.num_samples:
                for ns in v.num_samples:
                    ns = int(ns)
                    suffix = f"mc_ns{ns}"
                    slug = (v.name + "_" + suffix) if v.name else suffix
                    out.append(ExpandedVariant(slug, v, None, ns))
            if v.fractions:
                for frac in v.fractions:
                    f = float(frac)
                    suffix = f"mc_frac{str(f).replace('.', '_')}"
                    slug = (v.name + "_" + suffix) if v.name else suffix
                    out.append(ExpandedVariant(slug, v, f, None))
    return out


def _reinstantiate_mc(base: Explainer,
                      shap_cfg: ShapConfig,
                      *,
                      num_samples: int | None,
                      fraction: float | None) -> Explainer:
    """Re-create MC explainer with specific parameters (either num_samples or fraction)."""
    model = base.model
    mode = Mode[shap_cfg.mode]
    normalizer = NORMALIZER_MAP[shap_cfg.normalizer]()
    reducer = REDUCER_MAP[shap_cfg.reducer]()
    if num_samples is not None:
        mc = McShapExplainer(
            num_samples=int(num_samples),
            mode=mode,
            embedding_reducer=reducer,
            similarity_measure=CosineSimilarity(),
            normalizer=normalizer,
        )
    else:
        mc = McShapExplainer(
            num_samples=None,
            fraction=float(fraction) if fraction is not None else None,
            mode=mode,
            embedding_reducer=reducer,
            similarity_measure=CosineSimilarity(),
            normalizer=normalizer,
        )
    return Explainer(model=model, shap_explainer=mc)


def _reinstantiate_mc_for_num_samples(
    base: Explainer, shap_cfg: ShapConfig, num_samples: int
) -> Explainer:
    """Re-create MC explainer with a specific number of samples."""
    model = base.model
    mode = Mode[shap_cfg.mode]
    normalizer_cls = NORMALIZER_MAP[shap_cfg.normalizer]
    reducer_cls = REDUCER_MAP[shap_cfg.reducer]
    normalizer = normalizer_cls()
    reducer = reducer_cls()
    return Explainer(
        model=model,
        shap_explainer=McShapExplainer(
            num_samples=int(num_samples),
            mode=mode,
            embedding_reducer=reducer,
            similarity_measure=CosineSimilarity(),
            normalizer=normalizer,
        ),
    )


def run_single_sentence_variant(  # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    cfg: ExperimentSet,
    run: ExpandedVariant,
    df: pd.DataFrame,
    resume: bool,
) -> None:
    """
    Execute one concrete variant over the selected rows and persist all artifacts.
    """
    device = pick_device(cfg.device)
    run_dir = make_run_dir(cfg.output_root, cfg.experiment_set_id, run.run_slug)

    # ---- spec (saved once per variant)
    spec = {
        "experiment_set_id": cfg.experiment_set_id,
        "run_slug": run.run_slug,
        "variant": {
            "explainer_type": run.variant.explainer_type,
            "num_samples": run.num_samples,
            "fraction": run.fraction,
        },
        "dataset": cfg.dataset.__dict__,
        "selection": cfg.selection.__dict__,
        "generation": cfg.generation.__dict__,
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
        LOGGER.info("Resuming run '%s': %d completed samples found on disk.",
                    run.run_slug, len(ckpt["completed_indices"]))
    else:
        ckpt = {
            "completed_indices": [],
            "next_index": 0,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        save_json(ckpt_path, ckpt)

    # ---- W&B
    run_config = {**spec, "python_version": f"{torch.__version__}"}
    wb_run = wandb_init_if_enabled(
        cfg.wandb,
        run_name=f"{cfg.experiment_set_id}__{run.run_slug}",
        run_config=run_config,
    )

    # ---- explainer (build once, re-instantiate for specific MC params)
    explainer = build_explainer_for_variant(device, cfg.shap, run.variant, cfg.model_kind)
    if run.variant.explainer_type.lower() == ExplainerType.MC.value:
        explainer = _reinstantiate_mc(explainer, cfg.shap, num_samples=run.num_samples, fraction=run.fraction)

    # ---- data iteration
    text_col = choose_prompt_text_column(df)

    is_text_only = cfg.model_kind == ModelKind.TRANSFORMERS_TEXT.value
    working_df = df
    if is_text_only and cfg.selection.max_prompt_tokens is not None:
        tokenizer = get_hf_text_tokenizer()
        filtered_df, count = filter_df_by_max_prompt_tokens(
            df=working_df, text_col=text_col, tokenizer=tokenizer, max_tokens=int(cfg.selection.max_prompt_tokens)
        )
        LOGGER.info(
            "Prompt token filter: <= %d tokens matched %d / %d rows. Using filtered subset.",
            cfg.selection.max_prompt_tokens,
            count,
            len(working_df),
        )
        working_df = filtered_df

    for row_idx, row in iter_rows_for_selection(
        df=working_df,
        start_index=cfg.selection.start_index,
        max_samples=cfg.selection.max_samples,
        shuffle_seed=cfg.selection.shuffle_seed,
    ):
        # Skip if already completed (resume)
        if row_idx in set(ckpt["completed_indices"]):
            continue
        LOGGER.info("Running sample index %d for variant '%s'.", row_idx, run.run_slug)

        # Audio column resolution
        audio_bytes = None
        if not is_text_only:
            if AudioCol.MALE.value in row:
                audio_bytes = row[AudioCol.MALE.value][0]
            elif AudioCol.FEMALE.value in row:
                audio_bytes = row[AudioCol.FEMALE.value][0]
            else:
                raise KeyError("Expected 'audio__male' or 'audio__female' in row for audio model.")

        # Prompt resolution
        prompt_list = row[text_col]
        user_text = str(prompt_list[0])

        # Build chat
        chat = build_chat(explainer.model, user_text, audio_bytes, text_only=is_text_only)
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

        conv = result.full_chat.get_conversation()

        # summary + serialization
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
        save_json(run_dir / "samples" / f"sample_{row_idx:05d}_result.json", sample_result)

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

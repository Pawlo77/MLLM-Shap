"""Core orchestration: expand variants, execute runs via composable stages."""

import gc
import itertools
import json
import logging
import os
import time
from dataclasses import dataclass
from logging import Logger
from typing import Any, Dict, Iterator, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import torch
from mllm_shap.connectors.base.audio import SpectrogramGuidedAligner
from mllm_shap.shap.base._masks_manager import MasksManager
from mllm_shap.shap.base.approx import BaseShapApproximation
from mllm_shap.shap.hierarchical import HierarchicalExplainer
from mllm_shap.shap.neyman._base import BaseComplementaryNeymanShapExplainer

from .config import (
    ChatConfig,
    ExperimentSet,
    ExplainerVariant,
    HierarchicalConfig,
    RuntimeConfig,
)
from .constants import (
    ExplainerType,
    InputModality,
    MC_LIKE_EXPLAINERS,
    audio_column_for,
    is_text_only_modality,
)
from .data import (
    apply_filters,
    choose_prompt_text_column,
    extract_texts_from_row,
    iter_balanced_token_count_rows,
    iter_rows_for_selection,
)
from .factory import (
    build_chat,
    build_explainer_for_variant,
    build_generation_kwargs,
    build_token_filter,
)
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


# ---------------------------
# EXPANDED VARIANT
# ---------------------------


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
    hier_mode: Optional[str] = None


# ---------------------------
# VARIANT EXPANSION (Protocol-based)
# ---------------------------


def _expand_exact(v: ExplainerVariant) -> List[ExpandedVariant]:
    slug = v.name or ExplainerType.EXACT.value
    return [
        ExpandedVariant(
            run_slug=slug, variant=v, fraction=None, num_samples=None, linear=None
        )
    ]


def _expand_mc_like(v: ExplainerVariant) -> List[ExpandedVariant]:
    t = v.explainer_type
    out: List[ExpandedVariant] = []
    if v.num_samples:
        for ns in v.num_samples:
            ns_i = int(ns)
            suffix = f"{t}_ns{ns_i}"
            slug = f"{v.name}_{suffix}" if v.name else suffix
            out.append(ExpandedVariant(slug, v, None, ns_i, None))
    if v.fractions:
        for frac in v.fractions:
            f = float(frac)
            suffix = f"{t}_frac{str(f).replace('.', '_')}"
            slug = f"{v.name}_{suffix}" if v.name else suffix
            out.append(ExpandedVariant(slug, v, f, None, None))
    if v.linear:
        for lin in v.linear:
            lin_f = float(lin)
            suffix = f"{t}_lin{str(lin_f).replace('.', '_')}"
            slug = f"{v.name}_{suffix}" if v.name else suffix
            out.append(ExpandedVariant(slug, v, None, None, lin_f))
    return out


def _expand_hierarchical(v: ExplainerVariant) -> List[ExpandedVariant]:
    h = v.hierarchical or HierarchicalConfig()
    ks = h.ks or [10]
    inner_type = h.shap_type.lower()
    inner_ns_list = h.shap_num_samples or [None]
    inner_frac_list = h.shap_fractions or [None]
    fl_type = h.first_layer_type
    fl_ns_list = h.first_layer_num_samples or [None]
    fl_frac_list = h.first_layer_fractions or [None]
    imp_min_fracs = h.importance_min_fractions or [0.1]
    hier_mode = h.mode.value

    out: List[ExpandedVariant] = []

    for k, impmf, inn_ns, inn_fr in itertools.product(
        ks, imp_min_fracs, inner_ns_list, inner_frac_list
    ):
        if fl_type is None:
            slug = f"hier_{inner_type}_k{k}_imp{str(impmf).replace('.', '_')}"
            if inn_ns is not None:
                slug += f"_ns{inn_ns}"
            if inn_fr is not None:
                slug += f"_frac{str(inn_fr).replace('.', '_')}"
            out.append(
                ExpandedVariant(
                    run_slug=f"{v.name}_{slug}" if v.name else slug,
                    variant=v,
                    fraction=None,
                    num_samples=None,
                    linear=None,
                    hier_k=int(k),
                    hier_shap_type=inner_type,
                    hier_shap_num_samples=int(inn_ns) if inn_ns is not None else None,
                    hier_shap_fraction=float(inn_fr) if inn_fr is not None else None,
                    hier_first_layer_type=None,
                    hier_importance_min_fraction=float(impmf),
                    hier_mode=hier_mode,
                )
            )
        else:
            for flns, flfr in itertools.product(fl_ns_list, fl_frac_list):
                slug = f"hier_{inner_type}_k{k}_imp{str(impmf).replace('.', '_')}_fl{fl_type}"
                if inn_ns is not None:
                    slug += f"_ns{inn_ns}"
                if inn_fr is not None:
                    slug += f"_frac{str(inn_fr).replace('.', '_')}"
                if flns is not None:
                    slug += f"_flns{flns}"
                if flfr is not None:
                    slug += f"_flfrac{str(flfr).replace('.', '_')}"
                out.append(
                    ExpandedVariant(
                        run_slug=f"{v.name}_{slug}" if v.name else slug,
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
                        hier_first_layer_type=fl_type.lower(),
                        hier_first_layer_num_samples=int(flns)
                        if flns is not None
                        else None,
                        hier_first_layer_fraction=float(flfr)
                        if flfr is not None
                        else None,
                        hier_importance_min_fraction=float(impmf),
                        hier_mode=hier_mode,
                    )
                )
    return out


def expand_variants(cfg: ExperimentSet) -> List[ExpandedVariant]:
    """Materialize user-defined variants into concrete runs."""
    out: List[ExpandedVariant] = []
    for v in cfg.experiments:
        t = v.explainer_type
        if t == ExplainerType.EXACT:
            out.extend(_expand_exact(v))
        elif t in MC_LIKE_EXPLAINERS:
            out.extend(_expand_mc_like(v))
        elif t == ExplainerType.HIERARCHICAL:
            out.extend(_expand_hierarchical(v))
        else:
            raise ValueError(f"Unsupported explainer_type: {t}")
    return out


# ---------------------------
# LINEAR SAMPLE SCALER
# ---------------------------


@dataclass(frozen=True)
class _LinearSampleScaler:
    factor: float

    def scale(self, n_pre: int) -> int:
        scaled = int(self.factor * n_pre * n_pre)
        return scaled if scaled % 2 == 0 else scaled + 1


def _try_set_num_samples(explainer: Any, num_samples: int) -> bool:
    """Update sampling budget in-place for approximation explainers."""
    shap_explainer = getattr(explainer, "shap_explainer", None)
    if isinstance(shap_explainer, BaseShapApproximation):
        shap_explainer.num_samples = int(num_samples)
        return True
    return False


# ---------------------------
# STAGE: ROW SELECTOR
# ---------------------------


class RowSelector:
    """Selects and yields rows from the DataFrame based on config."""

    def __init__(self, cfg: ExperimentSet, df: pd.DataFrame) -> None:
        self._cfg = cfg
        self._df = df

    def iterate(self) -> Iterator[Tuple[int, Dict[str, Any]]]:
        sel = self._cfg.selection
        df = self._df

        # Apply generic filters
        if sel.filters:
            df = apply_filters(df, sel.filters)

        if sel.balanced_token_counts:
            token_col = self._cfg.dataset.column_mapping.token_count
            yield from iter_balanced_token_count_rows(
                df,
                token_counts=sel.balanced_token_counts,
                samples_per_token_count=int(sel.samples_per_token_count or 0),
                start_index=sel.start_index,
                max_samples=sel.max_samples,
                shuffle_seed=sel.shuffle_seed,
                allow_partial_buckets=sel.allow_partial_token_count_buckets,
                token_count_col=token_col,
            )
        else:
            yield from iter_rows_for_selection(
                df=df,
                start_index=sel.start_index,
                max_samples=None,  # Full scan; break on matched count
                shuffle_seed=sel.shuffle_seed,
            )


# ---------------------------
# STAGE: CHAT BUILDER
# ---------------------------


class ChatBuilder:
    """Builds chat instances from row data."""

    def __init__(
        self,
        model: Any,
        input_modality: InputModality,
        audio_segmentation_method: str,
        token_filter: Any,
        aligner: Optional[SpectrogramGuidedAligner],
        chat_cfg: ChatConfig,
    ) -> None:
        self._model = model
        self._input_modality = input_modality
        self._audio_seg = audio_segmentation_method
        self._token_filter = token_filter
        self._aligner = aligner
        self._chat_cfg = chat_cfg

    def build(
        self,
        user_texts: List[str],
        audio_bytes_list: Optional[List[bytes]],
    ) -> Any:
        return build_chat(
            self._model,
            user_texts=user_texts,
            audio_bytes_list=audio_bytes_list,
            input_modality=self._input_modality,
            token_filter=self._token_filter,
            audio_segmentation_method=self._audio_seg,
            aligner=self._aligner,
            chat_cfg=self._chat_cfg,
        )


# ---------------------------
# STAGE: EXPLAINER RUNNER
# ---------------------------


class ExplainerRunner:
    """Runs the explainer on a prepared chat and returns result + timing."""

    def __init__(self, explainer: Any, runtime_cfg: RuntimeConfig) -> None:
        self._explainer = explainer
        self._runtime = runtime_cfg

    @property
    def explainer(self) -> Any:
        return self._explainer

    def run(
        self, chat: Any, generation_kwargs: Dict[str, Any]
    ) -> Tuple[Any, float, int]:
        """Returns (result, runtime_sec, n_calls)."""
        t0 = time.time()
        result = self._explainer(
            chat=chat,
            verbose=self._runtime.verbose,
            generation_kwargs=generation_kwargs,
            progress_bar=self._runtime.progress_bar,
        )
        runtime_sec = time.time() - t0
        n_calls = self._explainer.total_n_calls
        return result, runtime_sec, n_calls


# ---------------------------
# STAGE: RESULT WRITER
# ---------------------------


class ResultWriter:
    """Serializes and persists sample results."""

    def __init__(self, run_dir: Any, wb_run: Any, cfg: ExperimentSet) -> None:
        self._run_dir = run_dir
        self._wb_run = wb_run
        self._cfg = cfg

    def save_sample(
        self,
        row_idx: int,
        row: Dict[str, Any],
        result: Any,
        runtime_sec: float,
        n_calls: int,
        user_texts: List[str],
        input_modality: InputModality,
        output_modality: Any,
        audio_bytes_list: Optional[List[bytes]],
        explainer: Any,
        run: ExpandedVariant,
    ) -> Dict[str, Any]:
        """Serialize a single sample result to disk and log to WandB."""
        conv = result.full_chat.get_conversation()
        modality_summary = compute_modality_summary(conv)
        conv_json = serialize_conversation(conv)

        lang_col = self._cfg.dataset.column_mapping.language
        orig_lang_col = self._cfg.dataset.column_mapping.original_language

        sample_result: Dict[str, Any] = {
            "row_index": int(row_idx),
            "language": row.get(lang_col, "unknown"),
            "original_language": row.get(orig_lang_col, "unknown"),
            "runtime_sec": float(runtime_sec),
            "n_calls": n_calls,
            "prompt_texts": user_texts,
            "input_modality": input_modality.value,
            "output_modality": output_modality.value,
            "attr_summary": modality_summary,
            "conversation": conv_json,
        }

        # Audio artifacts
        audio_artifacts_dir = self._run_dir / "audio" / f"sample_{row_idx:05d}"
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
            if self._wb_run is not None:
                log_audio_artifacts(
                    self._wb_run,
                    audio_artifacts_dir,
                    input_modality,
                    output_modality,
                    f"sample_{row_idx:05d}",
                )

        # Explainer-specific metadata
        if run.variant.explainer_type == ExplainerType.HIERARCHICAL:
            sample_result["num_levels"] = cast(HierarchicalExplainer, explainer).n_calls
        elif run.variant.explainer_type in (
            ExplainerType.LIMITED_NEYMAN,
            ExplainerType.STANDARD_NEYMAN,
        ):
            neyman_explainer = cast(
                BaseComplementaryNeymanShapExplainer, explainer.shap_explainer
            )
            sample_result["neyman_steps"] = (
                neyman_explainer._BaseComplementaryNeymanShapExplainer__step
            )

        sample_path = self._run_dir / "samples" / f"sample_{row_idx:05d}_result.json"
        save_json(sample_path, sample_result)

        if self._wb_run is not None:
            wandb_log_dir_incremental(
                self._wb_run,
                self._run_dir / "samples",
                artifact_name=f"{self._cfg.experiment_set_id}__{run.run_slug}-samples",
                artifact_type="samples",
                metadata={
                    "subset": self._cfg.dataset.subset,
                    "split": self._cfg.dataset.split,
                },
            )

        log_metrics(
            self._wb_run,
            {
                "progress/sample_index": row_idx,
                "timing/runtime_sec": runtime_sec,
                "attr/abs_sum_text": modality_summary["abs_sum_text"],
                "attr/abs_sum_audio": modality_summary["abs_sum_audio"],
                "attr/frac_text": modality_summary["frac_text"],
                "attr/frac_audio": modality_summary["frac_audio"],
                "counts/text_tokens": modality_summary["count_text_tokens"],
                "counts/audio_segments": modality_summary["count_audio_segments"],
                "meta/language": row.get(lang_col, "unknown"),
                "meta/input_modality": input_modality.value,
                "meta/output_modality": output_modality.value,
            },
        )

        return modality_summary

    def save_aggregate(self, run_slug: str) -> None:
        """Compute and save aggregate summary from all sample files."""
        summaries: List[Dict[str, Any]] = []
        for p in sorted((self._run_dir / "samples").glob("sample_*_result.json")):
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

        save_json(self._run_dir / "summary" / "aggregate_metrics.json", agg)

        if self._wb_run is not None:
            wandb_log_artifact(
                self._wb_run,
                self._run_dir / "summary" / "aggregate_metrics.json",
                artifact_name=f"{self._cfg.experiment_set_id}__{run_slug}-summary",
                artifact_type="summary",
                metadata={
                    "subset": self._cfg.dataset.subset,
                    "split": self._cfg.dataset.split,
                },
            )


# ---------------------------
# GPU CLEANUP
# ---------------------------


def _cleanup_gpu(result: Any, runtime_cfg: RuntimeConfig) -> None:
    """Aggressively free GPU memory after each sample."""
    if hasattr(result, "full_chat") and result.full_chat.cache is not None:
        result.full_chat.cache = None

    if result.history is not None:
        for mask, mask_hash, masked_chat, response in result.history:
            del mask, mask_hash, masked_chat, response
        del result.history

    del result

    if runtime_cfg.gc_after_each_sample:
        gc.collect()

    if runtime_cfg.cuda_empty_cache:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.synchronize()


# ---------------------------
# MAIN ORCHESTRATOR
# ---------------------------


def pick_device(name: Optional[str]) -> torch.device:
    """Resolve the torch device preference (cuda > mps > cpu)."""
    if name is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


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

    # Resolve effective configs (with per-variant overrides)
    effective_shap = cfg.get_effective_shap(run.variant)
    effective_gen = cfg.get_effective_generation(run.variant)
    effective_emb = cfg.get_effective_embedding(run.variant)

    # ---- spec (saved once per variant)
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
        "embedding": effective_emb.model_dump(),
        "audio_segmentation": cfg.audio_segmentation.model_dump(),
        "shap": effective_shap.model_dump(),
        "chat": cfg.chat.model_dump(),
        "runtime": cfg.runtime.model_dump(),
        "device": str(device),
    }
    save_json(run_dir / "spec.json", spec)

    # ---- checkpoint
    ckpt_path = run_dir / "checkpoint.json"
    disable_checkpoint = os.environ.get("MLLM_SHAPX_DISABLE_CHECKPOINT") == "1"
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
            "Resuming run '%s': %d completed samples.",
            run.run_slug,
            len(ckpt["completed_indices"]),
        )
    else:
        ckpt = {
            "version": 2,
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
    input_modality = cfg.modality.input_modality
    output_modality = cfg.modality.output_modality
    audio_segmentation_method = cfg.audio_segmentation.method

    # ---- explainer
    explainer = build_explainer_for_variant(
        device,
        effective_shap,
        run.variant,
        cfg.connector,
        embedding_cfg=effective_emb,
        output_modality=output_modality,
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

    # ---- stages
    token_filter = build_token_filter(effective_shap.token_filter)
    aligner = (
        SpectrogramGuidedAligner(
            device=torch.device(cfg.audio_segmentation.aligner_device)
        )
        if audio_segmentation_method == "sgpa"
        else None
    )

    text_col = choose_prompt_text_column(df, cfg.dataset.column_mapping.text)
    is_text = is_text_only_modality(input_modality)
    audio_col_name = audio_column_for(input_modality)

    row_selector = RowSelector(cfg, df)
    chat_builder = ChatBuilder(
        model=explainer.model,
        input_modality=input_modality,
        audio_segmentation_method=audio_segmentation_method,
        token_filter=token_filter,
        aligner=aligner,
        chat_cfg=cfg.chat,
    )
    explainer_runner = ExplainerRunner(explainer, cfg.runtime)
    result_writer = ResultWriter(run_dir, wb_run, cfg)
    generation_kwargs = build_generation_kwargs(effective_gen)

    min_t = cfg.selection.min_prompt_tokens
    max_t = cfg.selection.max_prompt_tokens

    # How many NEW matched samples we still need
    desired_total = cfg.selection.max_samples
    if desired_total is None:
        remaining_needed = float("inf")
    else:
        remaining_needed = max(int(desired_total) - len(completed_set), 0)

    if remaining_needed == 0:
        LOGGER.info(
            "Target max_samples=%s already satisfied. Nothing to do.",
            str(cfg.selection.max_samples),
        )

    # Counters
    scanned_ctr = 0
    ge_min_ctr = 0
    matched_ctr = 0
    runtimes_for_eta: List[float] = []

    for row_idx, row in row_selector.iterate():
        if matched_ctr >= remaining_needed:
            break
        if row_idx in completed_set:
            continue

        scanned_ctr += 1

        # ---- Resolve audio bytes
        audio_bytes_list: Optional[List[bytes]] = None
        if audio_col_name is not None:
            col_name = cfg.dataset.column_mapping.audio or audio_col_name.value
            if col_name in row:
                audio_bytes_list = row[col_name]
                if not isinstance(audio_bytes_list, (list, np.ndarray)):
                    audio_bytes_list = [audio_bytes_list]
            else:
                raise KeyError(
                    f"Expected '{col_name}' in row for {input_modality} input modality."
                )

        # ---- Prompt resolution
        user_texts = extract_texts_from_row(row[text_col])

        # ---- Build chat
        chat = chat_builder.build(
            user_texts=user_texts, audio_bytes_list=audio_bytes_list
        )

        # ---- Token counting
        mask = getattr(chat, "shap_values_mask", None)
        if mask is not None and hasattr(mask, "sum"):
            n_pre = int(mask.sum().item())
        else:
            n_pre = int(MasksManager(chat).n)

        if min_t is None or n_pre >= int(min_t):
            ge_min_ctr += 1

        # Apply bounds (text-only filtering)
        in_bounds = True
        if is_text:
            if (min_t is not None and n_pre < int(min_t)) or (
                max_t is not None and n_pre > int(max_t)
            ):
                in_bounds = False

        if not in_bounds:
            continue

        # ---- Matched row: run explainer
        LOGGER.info("Running sample index %d for variant '%s'.", row_idx, run.run_slug)
        LOGGER.info(" | ".join(user_texts))

        if run.linear:
            scaled_num_samples = _LinearSampleScaler(factor=run.linear).scale(
                n_pre=n_pre
            )
            LOGGER.info("Linear scaling: num_samples=%d.", scaled_num_samples)
            if not _try_set_num_samples(
                explainer=explainer, num_samples=scaled_num_samples
            ):
                explainer = build_explainer_for_variant(
                    device,
                    effective_shap,
                    run.variant,
                    cfg.connector,
                    embedding_cfg=effective_emb,
                    output_modality=output_modality,
                    connector_kwargs=cfg.connector_kwargs,
                    concrete_num_samples=scaled_num_samples,
                    concrete_fraction=run.fraction,
                )
                explainer_runner = ExplainerRunner(explainer, cfg.runtime)

        result, runtime_sec, n_calls = explainer_runner.run(chat, generation_kwargs)

        # Parity check
        n_post = int(MasksManager(getattr(result, "base_chat", chat)).n)
        if n_post != n_pre:
            LOGGER.warning(
                "Token drift: pre=%d post=%d (row=%d).", n_pre, n_post, row_idx
            )

        # ---- Serialize + metrics
        result_writer.save_sample(
            row_idx=row_idx,
            row=row,
            result=result,
            runtime_sec=runtime_sec,
            n_calls=n_calls,
            user_texts=user_texts,
            input_modality=input_modality,
            output_modality=output_modality,
            audio_bytes_list=audio_bytes_list,
            explainer=explainer,
            run=run,
        )

        # ---- Checkpoint
        if not disable_checkpoint:
            update_checkpoint(
                ckpt_path, ckpt, just_completed=row_idx, next_index=row_idx + 1
            )
        completed_set.add(row_idx)
        matched_ctr += 1

        # ---- ETA logging
        runtimes_for_eta.append(runtime_sec)
        if len(runtimes_for_eta) >= 3 and remaining_needed != float("inf"):
            avg_rt = np.mean(runtimes_for_eta[-10:])
            remaining = remaining_needed - matched_ctr
            eta_sec = avg_rt * remaining
            LOGGER.info(
                "Progress: %d/%d | ETA: %.0fs (avg %.1fs/sample)",
                matched_ctr,
                int(remaining_needed),
                eta_sec,
                avg_rt,
            )

        # ---- GPU cleanup
        _cleanup_gpu(result, cfg.runtime)

    # ---- Aggregate summary
    result_writer.save_aggregate(run.run_slug)

    # ---- Filter logs
    if is_text and min_t is not None:
        LOGGER.info(
            "Token filter: >= %d matched %d / %d rows.",
            int(min_t),
            ge_min_ctr,
            scanned_ctr,
        )
    if is_text and max_t is not None:
        denom = ge_min_ctr if min_t is not None else scanned_ctr
        LOGGER.info(
            "Token filter: <= %d matched %d / %d rows.", int(max_t), matched_ctr, denom
        )

    if wb_run is not None:
        wb_run.finish()

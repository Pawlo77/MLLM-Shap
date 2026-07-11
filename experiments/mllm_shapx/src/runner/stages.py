"""Runner stage components: row selection, chat build, execution, and result writing."""

import gc
import logging
import time
from typing import Any, Dict, Iterator, List, Mapping, Tuple, cast

import numpy as np
import pandas as pd
import torch
from mllm_shap.connectors.base.audio import SpectrogramGuidedAligner
from mllm_shap.shap.hierarchical import HierarchicalExplainer
from mllm_shap.shap.neyman._base import BaseComplementaryNeymanShapExplainer

from ..config import ChatConfig, ExperimentSet, RuntimeConfig
from ..constants import InputModality
from ..data import (
    apply_filters,
    iter_balanced_token_count_rows,
    iter_rows_for_selection,
)
from .types import ExpandedVariant

LOGGER = logging.getLogger(__name__)


class RowSelector:
    """Select rows according to config selection rules."""

    def __init__(self, cfg: ExperimentSet, df: pd.DataFrame) -> None:
        self._cfg = cfg
        self._df = df

    def iterate(self) -> Iterator[Tuple[int, Dict[str, Any]]]:
        """Yield selected rows as (row_idx, row_dict) pairs."""
        sel = self._cfg.selection
        df = self._df

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
                max_samples=None,
                shuffle_seed=sel.shuffle_seed,
            )


class ChatBuilder:
    """Build a chat object from selected row inputs."""

    def __init__(
        self,
        model: Any,
        input_modality: InputModality,
        audio_segmentation_method: str,
        token_filter: Any,
        aligner: SpectrogramGuidedAligner | None,
        chat_cfg: ChatConfig,
        build_chat_fn: Any,
    ) -> None:
        self._model = model
        self._input_modality = input_modality
        self._audio_seg = audio_segmentation_method
        self._token_filter = token_filter
        self._aligner = aligner
        self._chat_cfg = chat_cfg
        self._build_chat = build_chat_fn

    def build(self, user_texts: List[str], audio_bytes_list: List[bytes] | None) -> Any:
        """Construct a chat for the configured model and modalities."""
        return self._build_chat(
            self._model,
            user_texts=user_texts,
            audio_bytes_list=audio_bytes_list,
            input_modality=self._input_modality,
            token_filter=self._token_filter,
            audio_segmentation_method=self._audio_seg,
            aligner=self._aligner,
            chat_cfg=self._chat_cfg,
        )


class ExplainerRunner:
    """Execute an explainer call and report timing and call count."""

    def __init__(self, explainer: Any, runtime_cfg: RuntimeConfig) -> None:
        self._explainer = explainer
        self._runtime = runtime_cfg

    @property
    def explainer(self) -> Any:
        """Expose the underlying explainer wrapper."""
        return self._explainer

    def run(
        self, chat: Any, generation_kwargs: Dict[str, Any]
    ) -> Tuple[Any, float, int]:
        """Run explainer and return (result, runtime_sec, n_calls)."""
        t0 = time.time()
        result = self._explainer(
            chat=chat,
            verbose=self._runtime.verbose,
            generation_kwargs=generation_kwargs,
            progress_bar=self._runtime.progress_bar,
            n_generator_jobs=self._runtime.n_generator_jobs,
        )
        runtime_sec = time.time() - t0
        n_calls = self._explainer.total_n_calls
        return result, runtime_sec, n_calls


class ResultWriter:
    """Serialize sample-level and aggregate results to MLflow (no disk writes)."""

    def __init__(self, tracker: Any, cfg: ExperimentSet | Any) -> None:
        self._tracker = tracker
        self._cfg = cfg
        self._runtimes: List[float] = []
        self._frac_text: List[float] = []
        self._frac_audio: List[float] = []

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
        audio_bytes_list: List[bytes] | None,
        explainer: Any,
        run: ExpandedVariant,
        telemetry_metrics: Mapping[str, float] | None,
    ) -> Dict[str, Any]:
        """Log one sample result to MLflow as metrics and JSON artifact."""
        from . import (
            _compute_modality_summary,
            _serialize_conversation,
            serialize_result_with_audio,
        )

        conv = result.full_chat.get_conversation()
        modality_summary = _compute_modality_summary(conv)
        conv_json = _serialize_conversation(conv)

        lang_col = self._cfg.dataset.column_mapping.language
        orig_lang_col = self._cfg.dataset.column_mapping.original_language

        sample_result: Dict[str, Any] = {
            "result_schema_version": 2,
            "row_index": int(row_idx),
            "language": row.get(lang_col, "unknown"),
            "original_language": row.get(orig_lang_col, "unknown"),
            "runtime_sec": float(runtime_sec),
            "n_calls": int(n_calls),
            "prompt_texts": user_texts,
            "input_modality": str(input_modality.value),
            "output_modality": str(output_modality.value),
            "attr_summary": modality_summary,
            "conversation": conv_json,
        }

        # Audio serialization (to MLflow artifacts via tracker)
        audio_info = serialize_result_with_audio(
            result=result,
            tracker=self._tracker,
            input_audio_bytes=audio_bytes_list,
            input_modality=input_modality,
            output_modality=output_modality,
            sample_id=f"sample_{row_idx:05d}",
        )
        if audio_info.get("input_audio") or audio_info.get("output_audio"):
            sample_result["audio_artifacts"] = audio_info

        explainer_type = str(run.variant.explainer_type).lower()
        if explainer_type == "hierarchical":
            sample_result["num_levels"] = cast(HierarchicalExplainer, explainer).n_calls
        elif explainer_type in ("limited_neyman", "standard_neyman"):
            neyman = cast(
                BaseComplementaryNeymanShapExplainer, explainer.shap_explainer
            )
            sample_result["neyman_steps"] = (
                neyman._BaseComplementaryNeymanShapExplainer__step
            )

        # Log sample result as JSON artifact to MLflow
        self._tracker.log_dict(
            sample_result, artifact_file=f"samples/sample_{row_idx:05d}_result.json"
        )

        # Log metrics
        metrics: Dict[str, float] = {
            "progress/sample_index": float(row_idx),
            "timing/runtime_sec": float(runtime_sec),
            "attr/abs_sum_text": float(modality_summary["abs_sum_text"]),
            "attr/abs_sum_audio": float(modality_summary["abs_sum_audio"]),
            "attr/frac_text": float(modality_summary["frac_text"]),
            "attr/frac_audio": float(modality_summary["frac_audio"]),
            "counts/text_tokens": float(modality_summary["count_text_tokens"]),
            "counts/audio_segments": float(modality_summary["count_audio_segments"]),
        }
        if telemetry_metrics:
            metrics.update({k: float(v) for k, v in telemetry_metrics.items()})
        self._tracker.log_metrics(metrics, step=row_idx)

        # Track for aggregate computation
        self._runtimes.append(float(runtime_sec))
        self._frac_text.append(float(modality_summary["frac_text"]))
        self._frac_audio.append(float(modality_summary["frac_audio"]))

        return modality_summary

    def save_aggregate(self, run_slug: str) -> None:
        """Compute and log aggregate metrics to MLflow."""
        if self._runtimes:
            aggregate = {
                "n_samples": len(self._runtimes),
                "runtime_sec_total": float(sum(self._runtimes)),
                "runtime_sec_avg": float(np.mean(self._runtimes)),
                "avg_frac_text": float(np.mean(self._frac_text)),
                "avg_frac_audio": float(np.mean(self._frac_audio)),
            }
        else:
            aggregate = {
                "n_samples": 0,
                "runtime_sec_total": 0.0,
                "runtime_sec_avg": 0.0,
                "avg_frac_text": 0.0,
                "avg_frac_audio": 0.0,
            }

        self._tracker.log_dict(
            aggregate, artifact_file="summary/aggregate_metrics.json"
        )


def cleanup_gpu(result: Any, runtime_cfg: RuntimeConfig, sample_idx: int = 0) -> None:
    """Release intermediate tensors and caches after each sample.

    Args:
        result: Explainer result to clean up.
        runtime_cfg: Runtime configuration controlling cleanup behaviour.
        sample_idx: Current sample counter (used for periodic GC).
    """
    if (
        hasattr(result, "full_chat")
        and getattr(result.full_chat, "cache", None) is not None
    ):
        result.full_chat.cache = None

    if getattr(result, "history", None) is not None:
        for mask, mask_hash, masked_chat, response in result.history:
            del mask, mask_hash, masked_chat, response
        del result.history

    del result

    if runtime_cfg.gc_after_each_sample:
        interval = max(1, runtime_cfg.gc_interval)
        if sample_idx % interval == 0:
            gc.collect()

    if runtime_cfg.cuda_empty_cache:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
